#!/usr/bin/env python3

import math
import os
import argparse
import sys
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import intel_extension_for_pytorch as ipex

import conf
from tqdm import tqdm
from libmy import libpool, libdata

def train_step(model, optimizer, criterion, inputs, outputs):
    def closure():
        optimizer.zero_grad(set_to_none=True)
        prediction = model(*inputs, *outputs)
        loss = criterion(prediction, *outputs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        return loss.item()
    
    loss_val = optimizer.step(closure)
    return loss_val

def scheduler_ReduceLROnPlateau(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=2,
        threshold=0.005,    
        threshold_mode='rel' # Relative improvement
    )

def scheduler_BloatSimple(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.999999,
        patience=200000,
        threshold=0.005,    
        threshold_mode='rel' # Relative improvement
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
            # Loss is improving (going down)
            self.wait_count += 1
            if self.wait_count >= self.patience_up:
                self._scale_lr(self.factor_up)
                self.wait_count = 0
        else:
            # Loss got worse or stagnated (negative or zero improvement)
            self._scale_lr(self.factor_down)
            self.wait_count = 0

        self.prev_loss = current_loss

    def _scale_lr(self, factor):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= factor

class AggressiveParanoidScheduler:
    def __init__(self, optimizer, factor_up=10.0, factor_down=0.1, patience_up=2):
        self.optimizer = optimizer
        self.factor_up = factor_up
        self.factor_down = factor_down
        self.patience_up = patience_up
        self.max_step = [10000 for param_group in self.optimizer.param_groups]
        self.wait_count = 0
        self.prev_loss = None

    def step(self, current_loss):
        if self.prev_loss is None:
            self.prev_loss = current_loss
            return

        improvement = self.prev_loss - current_loss

        if improvement > 0:
            # Loss is improving (going down)
            self.wait_count += 1
            if self.wait_count >= self.patience_up:
                if self._each_lr_is_smaller([a/self.factor_up for a in self.max_step]):
                    self._scale_lr(self.factor_up)
                self.wait_count = 0
        else:
            # Loss got worse or stagnated (negative or zero improvement)
            self._scale_lr(self.factor_down)
            self.wait_count = 0
            self.max_step = [min(self.max_step[i], self.optimizer.param_groups[i]['lr']) for i in range(len(self.optimizer.param_groups))]

        self.prev_loss = current_loss

    def _scale_lr(self, factor):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= factor
    def _set_lr(self, values):
        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]['lr'] = values[i]
    def _each_lr_is_smaller(self, values):
        for i in range(len(self.optimizer.param_groups)):
            if (self.optimizer.param_groups[i]['lr'] > values[i]):
                return 0
            return 1


def scheduler_Agressive(optimizer):
    return AggressiveScheduler(
        optimizer, 
        factor_up=10.0, 
        factor_down=0.1, 
        patience_up=2
    )

def scheduler_AgressiveParanoid(optimizer):
    return AggressiveParanoidScheduler(
        optimizer, 
        factor_up=2, 
        factor_down=0.5, 
        patience_up=2
    )


class ElasticScheduler:
    def __init__(self, optimizer, factor_up=10.0, factor_down=0.1, patience_up=2, patience_penalty=2):
        self.optimizer = optimizer
        self.factor_up = factor_up
        self.factor_down = factor_down
        self.base_patience_up = patience_up
        self.current_patience_up = patience_up
        self.patience_penalty = patience_penalty
        self.max_patience = 50 # Critical constraint to prevent infinite stochastic stalls
        self.wait_count = 0
        self.prev_loss = None

    def step(self, current_loss):
        if self.prev_loss is None:
            self.prev_loss = current_loss
            return

        improvement = self.prev_loss - current_loss

        if improvement > 0:
            # Loss is improving (going down)
            self.wait_count += 1
            if self.wait_count >= self.current_patience_up:
                self._scale_lr(self.factor_up)
                self.wait_count = 0
                if self.current_patience_up > self.base_patience_up:
                    self.current_patience_up = max(self.base_patience_up, self.current_patience_up - 1)
        else:
            # Loss got worse or stagnated (negative or zero improvement)
            self._scale_lr(self.factor_down)
            self.wait_count = 0
            self.current_patience_up = min(self.max_patience, int(self.current_patience_up * self.patience_penalty))

        self.prev_loss = current_loss

    def _scale_lr(self, factor):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= factor







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
        loss = None
        if closure is not None:
            loss = closure()
        self.step_count += 1
        if self.step_count % self.acc_steps == 0:
            self.optimizer.step()
        return loss

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

def optimizer_Adam_accumulate(model, lr):
    accumulation_steps=4
    base_opt = torch.optim.Adam(model.parameters(), lr=accumulation_steps*lr)
    return AccumulatingOptimizer(base_opt, accumulation_steps=accumulation_steps)











class SpatialAccumulatingOptimizer(torch.optim.Optimizer):
    """
    Implements a Heun's Method (RK2) spatial integration step.
    Evaluates gradient at p, steps to p_temp, evaluates gradient at p_temp,
    deduces the global mean direction, normalizes it, and applies the strict scheduler step size.
    """
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("Spatial accumulation strictly requires a closure to evaluate multiple spatial coordinates.")

        with torch.enable_grad():
            loss1 = closure()

        p_original = []
        g1_list = []
        
        # Store initial parameter coordinates and the primary gradient (g1)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p_original.append(p.clone())
                g1_list.append(p.grad.clone())

        idx = 0
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                # Note: Assuming standard gradient descent for the probe step.
                p.add_(g1_list[idx], alpha=-lr)
                idx += 1

        with torch.enable_grad():
            loss2 = closure() # Generates g2 internally in p.grad

        idx = 0
        mean_dirs = []
        global_norm_sq = 0.0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g2 = p.grad.clone()
                g1 = g1_list[idx]
                mean_dir = (g1 + g2) / 2.0
                mean_dirs.append(mean_dir)
                global_norm_sq += torch.sum(mean_dir ** 2).item()
                idx += 1
        global_norm = (global_norm_sq ** 0.5) + 1e-12
        idx = 0
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                normalized_dir = mean_dirs[idx] / global_norm
                p.copy_(p_original[idx])
                p.add_(normalized_dir, alpha=-lr)
                idx += 1
        return loss1

def optimizer_Spatial_Accumulate(model, lr):
    return SpatialAccumulatingOptimizer(model.parameters(), lr=lr)






def optimizer_Adam(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr)


















class OptimalRK2Scheduler:
    """
    Solves high-dimensional kinematic fallacies via element-weighted structural consensus.
    Solves VRAM fragmentation via in-place tensor operations.
    Decouples geometric walls from environmental shocks.
    """
    def __init__(self, optimizer, factor_up=2.0, factor_down=0.5, 
                 patience_up=5, noise_beta=0.9, sigma_tolerance=2.0, reversal_threshold=-0.05):
        self.optimizer = optimizer
        self.factor_up = factor_up
        self.factor_down = factor_down
        self.patience_up = patience_up
        self.wait_count = 0
        
        # Statistical State
        self.noise_beta = noise_beta
        self.sigma_tolerance = sigma_tolerance
        self.loss_ema = None
        self.loss_var = 0.0
        self.best_loss = float('inf')
        
        # Kinematic State
        self.p_prev = None
        self.d_prev = None
        self.reversal_threshold = reversal_threshold
        self.initialized = False

    @torch.no_grad()
    def step(self, current_loss):
        if not self.initialized:
            self.p_prev = [[p.detach().clone() for p in g['params'] if p.requires_grad] 
                           for g in self.optimizer.param_groups]
            self.d_prev = [[torch.zeros_like(p) for p in g['params'] if p.requires_grad] 
                           for g in self.optimizer.param_groups]
            self.loss_ema = current_loss
            self.best_loss = current_loss
            self.initialized = True
            return

        # 1. Logarithmic Structural Consensus Kinematics
        log_weighted_cos_sim_sum = 0.0
        total_log_weights = 0.0
        
        for g_idx, group in enumerate(self.optimizer.param_groups):
            p_idx_tracked = 0
            for p in group['params']:
                if not p.requires_grad:
                    continue

                d_curr = p.detach() - self.p_prev[g_idx][p_idx_tracked]
                d_prev_tensor = self.d_prev[g_idx][p_idx_tracked]

                norm_c = torch.norm(d_curr).item()
                norm_p = torch.norm(d_prev_tensor).item()
                # Logarithmic dampening of parameter scale to balance massive layers vs bottlenecks
                elements = p.numel()
                log_weight = math.log10(elements + 1.0)

                if norm_c > 1e-12 and norm_p > 1e-12:
                    dot = torch.sum(d_curr * d_prev_tensor).item()
                    cos_sim = dot / (norm_c * norm_p)
                    log_weighted_cos_sim_sum += cos_sim * log_weight
                
                total_log_weights += log_weight

                # O(1) Memory Update
                self.d_prev[g_idx][p_idx_tracked].copy_(d_curr)
                self.p_prev[g_idx][p_idx_tracked].copy_(p.detach())
                p_idx_tracked += 1

        mean_cos_sim = log_weighted_cos_sim_sum / total_log_weights if total_log_weights > 0 else 0.0
        is_reversal = mean_cos_sim < self.reversal_threshold

        # 2. Update Statistical Geometry
        self.loss_var = self.noise_beta * self.loss_var + (1 - self.noise_beta) * (current_loss - self.loss_ema)**2
        self.loss_ema = self.noise_beta * self.loss_ema + (1 - self.noise_beta) * current_loss
        sigma = (self.loss_var + 1e-8) ** 0.5
        noise_ceiling = self.loss_ema + self.sigma_tolerance * sigma

        # 3. Decoupled State Machine
        if current_loss < self.best_loss and not is_reversal:
            self.best_loss = current_loss
            self.wait_count += 1
            if self.wait_count >= self.patience_up:
                self._scale_lr(self.factor_up)
                self.wait_count = 0

        elif current_loss <= noise_ceiling and not is_reversal:
            pass

        else:
            if is_reversal:
                self._scale_lr(self.factor_down)
                self.wait_count = 0
            elif current_loss > noise_ceiling:
                self.wait_count = 0
                self.best_loss = current_loss
                self.loss_ema = current_loss
                self.loss_var = self.loss_var * 2.0 

    def _scale_lr(self, factor):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= factor






























class BoundedOptimalRK2LARSOptimizer(torch.optim.Optimizer):
    """
    Synthesizes RK2 with Bounded LARS.
    Prevents global scale tyranny while restoring native gradient deceleration 
    within the terminal braking radius to guarantee convergence.
    """
    def __init__(self, params, lr=1e-3, weight_decay=0.0, trust_coefficient=0.001, rho=1e-2, eps=1e-8):
        defaults = dict(lr=lr, weight_decay=weight_decay, trust_coefficient=trust_coefficient, rho=rho, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("Spatial accumulation strictly requires a closure.")

        with torch.enable_grad():
            loss1 = closure()

        p_original = []
        g1_list = []

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p_original.append(p.clone())
                g1_list.append(p.grad.clone())

        # Phase 2: Bounded RK2-LARS Probe Step
        idx = 0
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            trust_coefficient = group['trust_coefficient']
            rho = group['rho']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                g1 = g1_list[idx]
                if weight_decay != 0:
                    g1 = g1 + weight_decay * p.detach()

                # eps added to norm_p to prevent Zero-Initialization fractures
                norm_g1 = torch.norm(g1).item()
                
                # Bounded LARS Ratio: Enforces spatial normalization until ||g|| falls below rho.
                # When ||g|| < rho, the denominator locks at rho, and local_lr decreases as ||g|| decreases.
                scaling_factor = max(norm_g1, rho)
                is_excluded = p.ndim < 2
                if is_excluded:
                    # Fallback to Bounded RK2:
                    # Trust coefficient is preserved to prevent 1000x Kinetic Shear,
                    # but the unstable ||p|| multiplier is removed.
                    local_lr = lr * trust_coefficient / scaling_factor
                else:
                    norm_p = torch.norm(p.detach()).item() + eps
                    local_lr = lr * trust_coefficient * (norm_p / scaling_factor)

                p.add_(g1, alpha=-local_lr)
                idx += 1

        with torch.enable_grad():
            loss2 = closure() 

        # Phase 4: Bounded RK2-LARS Final Step
        idx = 0
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            trust_coefficient = group['trust_coefficient']
            rho = group['rho']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                g2 = p.grad.clone()
                g1 = g1_list[idx]
                
                mean_dir = (g1 + g2) / 2.0
                
                is_excluded = p.ndim < 2
                if not is_excluded and weight_decay != 0:
                    mean_dir = mean_dir + weight_decay * p_original[idx]


                
                norm_mean = torch.norm(mean_dir).item()
                
                scaling_factor = max(norm_mean, rho)
                
                if is_excluded:
                    # Maintain kinetic symmetry for 1D tensors
                    local_lr = lr * trust_coefficient / scaling_factor
                else:
                    norm_p = torch.norm(p_original[idx]).item() + eps
                    local_lr = lr * trust_coefficient * (norm_p / scaling_factor)
                
                p.copy_(p_original[idx])
                p.add_(mean_dir, alpha=-local_lr)
                idx += 1
                
        return loss1


def optimizer_Bounded_Optimal_RK2_LARS(model, lr):
    return BoundedOptimalRK2LARSOptimizer(model.parameters(), lr=lr)























def optimizer_LBFGS_Sniper(model, lr):
    return torch.optim.LBFGS(
        model.parameters(),
        lr=lr,               # This is now strictly controlled by the custom scheduler
        history_size=10,
        max_iter=1,          # CRITICAL: Force exactly one step per closure evaluation
        max_eval=1,          # CRITICAL: Prevent internal re-evaluations
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn=None  # CRITICAL: Disable safety checks. Accept the scheduler's step size blindly.
    )

def optimizer_LBFGS_Sniper_Wolfe(model, lr):
    return torch.optim.LBFGS(
        model.parameters(),
        lr=lr,               # Controlled by AggressiveParanoidScheduler
        history_size=10,
        max_iter=5,          # Bandwidth for matrix updates
        max_eval=10,         # Bandwidth for the line search to test the scheduler's extreme steps
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe" # The critical containment field
    )

def optimizer_LBFGS(model, lr):
    return torch.optim.LBFGS(
        model.parameters(), 
        lr=lr, 
        history_size=10, 
        max_iter=4,        # Limit internal iterations to cap overhead
        #line_search_fn="strong_wolfe" # NECESSARY for the "basin" stability you requested
    )


def criterion_MSELoss():
    return nn.MSELoss(reduction='sum') # Standard loss for regression (measurements)

def my_train(dataset, ModelClass, scheduler_fn, optimizer_fn, criterion_fn, verbose=False):
    epochs = 100
    libdata.norm_data_mean_stddev_len(dataset)
    device = torch.device("xpu") if torch.xpu.is_available() else torch.device("cpu")
    print(f"Training on: {device}")

    model = ModelClass(dataset_stats=dataset.stats)
    model = model.to(device)
    optimizer = optimizer_fn(model=model, lr=1e-4)

    if isinstance(optimizer, AccumulatingOptimizer):
        model, optimized_base = ipex.optimize(model, optimizer=optimizer.optimizer)
        optimizer.optimizer = optimized_base
    elif isinstance(optimizer, torch.optim.LBFGS):
        pass
        #model = ipex.optimize(model)
        #model = torch.compile(model, backend="ipex")
    else:
        model, optimizer = ipex.optimize(model, optimizer=optimizer)

    #model = torch.compile(model, backend="ipex")
    scheduler = scheduler_fn(optimizer)
    criterion = criterion_fn()
    print("is instance", isinstance(optimizer, torch.optim.LBFGS))
    batch_size = len(dataset) if isinstance(optimizer, torch.optim.LBFGS) else 256
    shuffle = False if isinstance(optimizer, torch.optim.LBFGS) else True 
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
    output_fullname = os.path.join(str(conf.OUTPUT_DIR), str(ModelClass.__name__)+"_weights.pth")

    if verbose:
        print("model name:", str(ModelClass.__name__))
        print("output full name:", str(output_fullname))
    model.train()
    prev_loss = 1
    for epoch in range(epochs):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        for traj_tuple in loop:
            inputs = [x.to(device, non_blocking=True) for x in traj_tuple["inputs"]]
            outputs = [x.to(device, non_blocking=True) for x in traj_tuple["outputs"]]
            loss_val = train_step(model=model, optimizer=optimizer, criterion=criterion,
                              inputs=inputs, outputs=outputs)
            total_loss += loss_val
            loop.set_postfix(loss=loss_val)
        current_lr = optimizer.param_groups[0]['lr']
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f} | Loss improvement: {(prev_loss-avg_loss)/prev_loss*100:.4f}% | Current LR: {current_lr:.2e}")
        scheduler.step(avg_loss)
        print(f"New LR: {optimizer.param_groups[0]['lr']:.2e}")
        prev_loss = avg_loss
    model.denormalize_weights()
    torch.save(model.state_dict(), output_fullname)


def main():
    parser = argparse.ArgumentParser(
        description="AI trainer",
        epilog="Examples: ./ai_train.py -v --dataset=StraightTrack --model=MyLSTM"
    )
    parser.add_argument(
        '-d', '--dataset',      # Aliases
        type=str,            # Formal type enforcement
        help='Class name of training dataset',
        required=True,       # Critical constraint: fails if missing
        metavar='FILE'       # Placeholder name in help text
    )
    parser.add_argument(
        '-m', '--model',      # Aliases
        type=str,            # Formal type enforcement
        help='Class name of model to be trained',
        required=True,       # Critical constraint: fails if missing
        metavar='FILE'       # Placeholder name in help text
    )
    parser.add_argument(
        '-v', '--verbose',   # Aliases
        action='store_true', # Takes NO sub-arguments
        help='Enable verbose output'
    )
    parser.add_argument(
        '-f', '--files',
        nargs='+',           # Greedily consumes remaining args
        help='List of specific python files containing classes'
    )
    try:
        args = parser.parse_args()
    except SystemExit:
        sys.exit(1)
    file_pool = libpool.get_file_pool(args.files, verbose=args.verbose)
    if args.verbose:
        print(f"Dataset class:    {args.dataset}")
        print(f"Model class:    {args.model}")
        print(f"File Pool: {file_pool}")
    
    try:
        DatasetClass, ds_module = libpool.load_class_from_pool(args.dataset, file_pool, verbose=args.verbose)
        ModelClass, _ = libpool.load_class_from_pool(args.model, file_pool, verbose=args.verbose)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    torch.set_num_threads(8)

    print(f"Instantiating Dataset: {DatasetClass.__name__}")
    dataset = DatasetClass(ModelClass.IO_CONFIG)
    train_data, val_data = libdata.my_train_val_split(dataset, 2/3)
    print(f"Starting training with model: {ModelClass.__name__}")
    #my_train(train_data, ModelClass, OptimalRK2Scheduler, optimizer_Bounded_Optimal_RK2_LARS, criterion_MSELoss, verbose=args.verbose)
    my_train(train_data, ModelClass, scheduler_BloatSimple, optimizer_Adam, criterion_MSELoss, verbose=args.verbose)

if __name__ == '__main__':
    main()

