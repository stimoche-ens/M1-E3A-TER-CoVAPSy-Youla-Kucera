#!/usr/bin/env python3

import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import conf
from libmy import libdata
from training_algorithms import maybe_ipex_optimize


def train_step(model, optimizer, criterion, inputs, outputs):
    def closure():
        optimizer.zero_grad(set_to_none=True)
        prediction = model(*inputs, *outputs)
        loss = criterion(prediction, *outputs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        return loss.item()

    return optimizer.step(closure)


def my_train(
    dataset,
    ModelClass,
    scheduler_fn,
    optimizer_fn,
    criterion_fn,
    verbose=False,
    use_ipex=True,
    output_path=None,
    epochs=100,
):
    libdata.norm_data_mean_stddev_len(dataset)
    device = torch.device("xpu") if torch.xpu.is_available() else torch.device("cpu")
    print(f"Training on: {device}")

    model = ModelClass(dataset_stats=dataset.stats).to(device)
    optimizer = optimizer_fn(model=model, lr=1e-4)
    model, optimizer = maybe_ipex_optimize(model, optimizer, use_ipex=use_ipex)
    scheduler = scheduler_fn(optimizer)
    criterion = criterion_fn()

    batch_size = len(dataset) if isinstance(optimizer, torch.optim.LBFGS) else 256
    shuffle = False if isinstance(optimizer, torch.optim.LBFGS) else True
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
    output_fullname = str(output_path or conf.default_weights_path(ModelClass.__name__))
    os.makedirs(os.path.dirname(output_fullname), exist_ok=True)

    if verbose:
        print("model name:", str(ModelClass.__name__))
        print("output full name:", output_fullname)

    model.train()
    prev_loss = 1
    for epoch in range(epochs):
        loop = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        for traj_tuple in loop:
            inputs = [x.to(device, non_blocking=True) for x in traj_tuple["inputs"]]
            outputs = [x.to(device, non_blocking=True) for x in traj_tuple["outputs"]]
            loss_val = train_step(model, optimizer, criterion, inputs, outputs)
            total_loss += loss_val
            loop.set_postfix(loss=loss_val)
        current_lr = optimizer.param_groups[0]["lr"]
        avg_loss = total_loss / len(loader)
        improvement = (prev_loss - avg_loss) / prev_loss * 100
        print(
            f"Epoch {epoch + 1} Average Loss: {avg_loss:.6f} | "
            f"Loss improvement: {improvement:.4f}% | Current LR: {current_lr:.2e}"
        )
        scheduler.step(avg_loss)
        print(f"New LR: {optimizer.param_groups[0]['lr']:.2e}")
        prev_loss = avg_loss

    model.denormalize_weights()
    torch.save(model.state_dict(), output_fullname)
    return output_fullname
