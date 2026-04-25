#!/usr/bin/env python3

import math

import torch
import torch.nn as nn


def scheduler_ReduceLROnPlateau(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=2,
        threshold=0.005,
        threshold_mode="rel",
    )


def scheduler_BloatSimple(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.999999,
        patience=200000,
        threshold=0.005,
        threshold_mode="rel",
    )


class AggressiveScheduler:
    def __init__(self, optimizer, factor_up=10.0, factor_down=0.1, patience_up=2):
        self.optimizer = optimizer
        self.factor_up = factor_up
        self.factor_down = factor_down
        self.patience_up = patience_up
        self.wait_count = 0
        self.prev_loss = None

    def step(self, current_loss):
        if self.prev_loss is None:
            self.prev_loss = current_loss
            return
        improvement = self.prev_loss - current_loss
        if improvement > 0:
            self.wait_count += 1
            if self.wait_count >= self.patience_up:
                self._scale_lr(self.factor_up)
                self.wait_count = 0
        else:
            self._scale_lr(self.factor_down)
            self.wait_count = 0
        self.prev_loss = current_loss

    def _scale_lr(self, factor):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= factor


class AggressiveParanoidScheduler(AggressiveScheduler):
    def __init__(self, optimizer, factor_up=10.0, factor_down=0.1, patience_up=2):
        super().__init__(optimizer, factor_up, factor_down, patience_up)
        self.max_step = [10000 for _ in self.optimizer.param_groups]

    def step(self, current_loss):
        if self.prev_loss is None:
            self.prev_loss = current_loss
            return
        improvement = self.prev_loss - current_loss
        if improvement > 0:
            self.wait_count += 1
            if self.wait_count >= self.patience_up:
                if self._each_lr_is_smaller([a / self.factor_up for a in self.max_step]):
                    self._scale_lr(self.factor_up)
                self.wait_count = 0
        else:
            self._scale_lr(self.factor_down)
            self.wait_count = 0
            self.max_step = [
                min(self.max_step[i], self.optimizer.param_groups[i]["lr"])
                for i in range(len(self.optimizer.param_groups))
            ]
        self.prev_loss = current_loss

    def _each_lr_is_smaller(self, values):
        return all(
            self.optimizer.param_groups[i]["lr"] <= values[i]
            for i in range(len(self.optimizer.param_groups))
        )


def scheduler_Agressive(optimizer):
    return AggressiveScheduler(optimizer, factor_up=10.0, factor_down=0.1, patience_up=2)


def scheduler_AgressiveParanoid(optimizer):
    return AggressiveParanoidScheduler(optimizer, factor_up=2, factor_down=0.5, patience_up=2)


class ElasticScheduler(AggressiveScheduler):
    def __init__(self, optimizer, factor_up=10.0, factor_down=0.1, patience_up=2, patience_penalty=2):
        super().__init__(optimizer, factor_up, factor_down, patience_up)
        self.base_patience_up = patience_up
        self.current_patience_up = patience_up
        self.patience_penalty = patience_penalty
        self.max_patience = 50

    def step(self, current_loss):
        if self.prev_loss is None:
            self.prev_loss = current_loss
            return
        improvement = self.prev_loss - current_loss
        if improvement > 0:
            self.wait_count += 1
            if self.wait_count >= self.current_patience_up:
                self._scale_lr(self.factor_up)
                self.wait_count = 0
                self.current_patience_up = max(self.base_patience_up, self.current_patience_up - 1)
        else:
            self._scale_lr(self.factor_down)
            self.wait_count = 0
            self.current_patience_up = min(
                self.max_patience,
                int(self.current_patience_up * self.patience_penalty),
            )
        self.prev_loss = current_loss


class OptimalRK2Scheduler:
    def __init__(
        self,
        optimizer,
        factor_up=2.0,
        factor_down=0.5,
        patience_up=5,
        noise_beta=0.9,
        sigma_tolerance=2.0,
        reversal_threshold=-0.05,
    ):
        self.optimizer = optimizer
        self.factor_up = factor_up
        self.factor_down = factor_down
        self.patience_up = patience_up
        self.wait_count = 0
        self.noise_beta = noise_beta
        self.sigma_tolerance = sigma_tolerance
        self.loss_ema = None
        self.loss_var = 0.0
        self.best_loss = float("inf")
        self.p_prev = None
        self.d_prev = None
        self.reversal_threshold = reversal_threshold
        self.initialized = False

    @torch.no_grad()
    def step(self, current_loss):
        if not self.initialized:
            self.p_prev = [
                [p.detach().clone() for p in group["params"] if p.requires_grad]
                for group in self.optimizer.param_groups
            ]
            self.d_prev = [
                [torch.zeros_like(p) for p in group["params"] if p.requires_grad]
                for group in self.optimizer.param_groups
            ]
            self.loss_ema = current_loss
            self.best_loss = current_loss
            self.initialized = True
            return

        weighted_cos_sum = 0.0
        total_weight = 0.0
        for group_index, group in enumerate(self.optimizer.param_groups):
            tracked_index = 0
            for parameter in group["params"]:
                if not parameter.requires_grad:
                    continue
                d_curr = parameter.detach() - self.p_prev[group_index][tracked_index]
                d_prev = self.d_prev[group_index][tracked_index]
                norm_c = torch.norm(d_curr).item()
                norm_p = torch.norm(d_prev).item()
                weight = math.log10(parameter.numel() + 1.0)
                if norm_c > 1e-12 and norm_p > 1e-12:
                    weighted_cos_sum += torch.sum(d_curr * d_prev).item() / (norm_c * norm_p) * weight
                total_weight += weight
                self.d_prev[group_index][tracked_index].copy_(d_curr)
                self.p_prev[group_index][tracked_index].copy_(parameter.detach())
                tracked_index += 1

        mean_cos = weighted_cos_sum / total_weight if total_weight > 0 else 0.0
        is_reversal = mean_cos < self.reversal_threshold
        self.loss_var = self.noise_beta * self.loss_var + (1 - self.noise_beta) * (current_loss - self.loss_ema) ** 2
        self.loss_ema = self.noise_beta * self.loss_ema + (1 - self.noise_beta) * current_loss
        noise_ceiling = self.loss_ema + self.sigma_tolerance * ((self.loss_var + 1e-8) ** 0.5)

        if current_loss < self.best_loss and not is_reversal:
            self.best_loss = current_loss
            self.wait_count += 1
            if self.wait_count >= self.patience_up:
                self._scale_lr(self.factor_up)
                self.wait_count = 0
        elif current_loss > noise_ceiling or is_reversal:
            self._scale_lr(self.factor_down)
            self.wait_count = 0
            self.best_loss = current_loss

    def _scale_lr(self, factor):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= factor


class AccumulatingOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer, accumulation_steps=1):
        self.optimizer = optimizer
        self.acc_steps = accumulation_steps
        self.step_count = 0
        self.param_groups = optimizer.param_groups
        self.defaults = optimizer.defaults
        self.state = optimizer.state

    def zero_grad(self, set_to_none=True):
        if self.step_count % self.acc_steps == 0:
            self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        self.step_count += 1
        if self.step_count % self.acc_steps == 0:
            self.optimizer.step()
        return loss

    def __getattr__(self, name):
        return getattr(self.optimizer, name)


def optimizer_Adam_accumulate(model, lr):
    accumulation_steps = 4
    base_opt = torch.optim.Adam(model.parameters(), lr=accumulation_steps * lr)
    return AccumulatingOptimizer(base_opt, accumulation_steps=accumulation_steps)


class SpatialAccumulatingOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("Spatial accumulation requires a closure.")
        with torch.enable_grad():
            loss1 = closure()
        originals = []
        g1_list = []
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                originals.append(parameter.clone())
                g1_list.append(parameter.grad.clone())
        idx = 0
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                parameter.add_(g1_list[idx], alpha=-group["lr"])
                idx += 1
        with torch.enable_grad():
            closure()
        idx = 0
        mean_dirs = []
        global_norm_sq = 0.0
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                mean_dir = (g1_list[idx] + parameter.grad.clone()) / 2.0
                mean_dirs.append(mean_dir)
                global_norm_sq += torch.sum(mean_dir ** 2).item()
                idx += 1
        global_norm = global_norm_sq ** 0.5 + 1e-12
        idx = 0
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                parameter.copy_(originals[idx])
                parameter.add_(mean_dirs[idx] / global_norm, alpha=-group["lr"])
                idx += 1
        return loss1


def optimizer_Spatial_Accumulate(model, lr):
    return SpatialAccumulatingOptimizer(model.parameters(), lr=lr)


def optimizer_Adam(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr)


class BoundedOptimalRK2LARSOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.0, trust_coefficient=0.001, rho=1e-2, eps=1e-8):
        defaults = dict(lr=lr, weight_decay=weight_decay, trust_coefficient=trust_coefficient, rho=rho, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("Bounded RK2-LARS requires a closure.")
        with torch.enable_grad():
            loss1 = closure()
        originals = []
        g1_list = []
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                originals.append(parameter.clone())
                g1_list.append(parameter.grad.clone())
        self._probe_step(g1_list)
        with torch.enable_grad():
            closure()
        self._final_step(originals, g1_list)
        return loss1

    def _local_lr(self, group, parameter, direction):
        norm_direction = torch.norm(direction).item()
        scale = max(norm_direction, group["rho"])
        if parameter.ndim < 2:
            return group["lr"] * group["trust_coefficient"] / scale
        norm_p = torch.norm(parameter.detach()).item() + group["eps"]
        return group["lr"] * group["trust_coefficient"] * (norm_p / scale)

    def _probe_step(self, g1_list):
        idx = 0
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                direction = g1_list[idx]
                if group["weight_decay"] != 0:
                    direction = direction + group["weight_decay"] * parameter.detach()
                parameter.add_(direction, alpha=-self._local_lr(group, parameter, direction))
                idx += 1

    def _final_step(self, originals, g1_list):
        idx = 0
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                direction = (g1_list[idx] + parameter.grad.clone()) / 2.0
                if parameter.ndim >= 2 and group["weight_decay"] != 0:
                    direction = direction + group["weight_decay"] * originals[idx]
                parameter.copy_(originals[idx])
                parameter.add_(direction, alpha=-self._local_lr(group, parameter, direction))
                idx += 1


def optimizer_Bounded_Optimal_RK2_LARS(model, lr):
    return BoundedOptimalRK2LARSOptimizer(model.parameters(), lr=lr)


def optimizer_LBFGS_Sniper(model, lr):
    return torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        history_size=10,
        max_iter=1,
        max_eval=1,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn=None,
    )


def optimizer_LBFGS_Sniper_Wolfe(model, lr):
    return torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        history_size=10,
        max_iter=5,
        max_eval=10,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )


def optimizer_LBFGS(model, lr):
    return torch.optim.LBFGS(model.parameters(), lr=lr, history_size=10, max_iter=4)


def criterion_MSELoss():
    return nn.MSELoss(reduction="sum")


def maybe_ipex_optimize(model, optimizer, use_ipex=True):
    if not use_ipex:
        return model, optimizer
    try:
        import intel_extension_for_pytorch as ipex
    except ImportError:
        print("IPEX not available; continuing without IPEX optimization.")
        return model, optimizer
    if isinstance(optimizer, AccumulatingOptimizer):
        model, optimized_base = ipex.optimize(model, optimizer=optimizer.optimizer)
        optimizer.optimizer = optimized_base
        return model, optimizer
    if isinstance(optimizer, torch.optim.LBFGS):
        return model, optimizer
    return ipex.optimize(model, optimizer=optimizer)
