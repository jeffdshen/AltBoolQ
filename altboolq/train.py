import copy
import json
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import DataLoader

from transformers import AutoTokenizer


from .utils import set_environ, set_seed
from .datasets import BoolQDataset, split_batch, score, to_device
from .models import BoolQModel, get_linear_warmup_power_decay_scheduler
from .stats import EMAMeter, AverageMeter, MaxMeter, AccMeter, AccEMAMeter


def forward_backward(model, example, model_batch_size, device, backward):
    x_batch, y_batch, labels = example
    x_chunks = split_batch(x_batch, model_batch_size)
    y_chunks = torch.split(y_batch, model_batch_size)
    z_chunks = []
    total_loss = []
    for x, y in zip(x_chunks, y_chunks):
        z = model(x)
        z_chunks.append(z.detach().clone())

        with torch.autocast(device.type):
            loss = model.get_loss(z, y, x)

        total_loss.append(loss.item())
        backward(loss)

    z_batch = torch.cat(z_chunks)

    pred = model.get_pred(z_batch, x_batch)
    scores = score(pred, labels)
    return np.average(total_loss), scores


def noop_backward(_):
    pass


def evaluate(model, device, loader, config):
    valid_loss_meter = AverageMeter()
    valid_acc_meter = AccMeter()

    model.eval()
    with torch.no_grad():
        for example in loader:
            example = to_device(example, device)
            batch_size, example = example[0], example[1:]

            loss, scores = forward_backward(
                model, example, config["model_batch_size"], device, noop_backward
            )

            valid_loss_meter.add(loss, batch_size)
            valid_acc_meter.add(scores, batch_size)

    return valid_loss_meter.avg, valid_acc_meter.acc


def train_loop(
    model,
    device,
    optimizer,
    scheduler,
    scaler,
    train_loader,
    valid_loader,
    config,
    wandb,
):
    step_num = 0
    samples_since_eval = 0
    sample_num = 0

    train_loss_meter = EMAMeter(config["train_loss_ema"])
    train_acc_meter = AccEMAMeter(config["train_loss_ema"])
    best_meter = MaxMeter()

    def backward(loss):
        nonlocal step_num
        # Backward
        scaler.scale(loss / config["gradient_accumulation"]).backward()

        # Step
        if (step_num + 1) % config["gradient_accumulation"] == 0:
            scaler.unscale_(optimizer)
            if config["max_grad_norm"] is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        step_num += 1

    for epoch in range(config["num_epochs"]):
        print(f"Starting epoch {epoch}")
        for example in train_loader:
            example = to_device(example, device)
            batch_size, example = example[0], example[1:]

            # Forward
            loss, scores = forward_backward(
                model, example, config["model_batch_size"], device, backward
            )
            sample_num += batch_size
            samples_since_eval += batch_size

            # Stats
            train_loss_meter.add(loss, batch_size)
            train_acc_meter.add(scores, batch_size)

            # Validation
            if samples_since_eval >= config["eval_per_n_samples"]:
                samples_since_eval = 0
                valid_loss, valid_acc = evaluate(model, device, valid_loader, config)
                is_best = best_meter.add(valid_acc)
                best = " (Best)" if is_best else ""

                results_dict = {
                    "epoch": epoch,
                    "sample": sample_num,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_loss": train_loss_meter.avg,
                    "valid_loss": valid_loss,
                    "train_error": train_acc_meter.acc,
                    "valid_error": valid_acc,
                }
                results_format = {
                    "epoch": "{:2d}",
                    "sample": "{:6d}",
                    "lr": "{:8.2e}",
                    "train_loss": "{:7.4f}",
                    "valid_loss": "{:7.4f}",
                    "train_error": "{:7.4f}",
                    "valid_error": "{:7.4f}",
                }
                results = [
                    "{}: {}".format(key, value).format(results_dict[key])
                    for key, value in results_format
                ]
                print(" | ".join(results) + best)
                wandb.log(results_dict)
                if is_best:
                    torch.save(model.state_dict(), f"./best.pth")
    return best_meter.max


def get_boolq_dataset(df, tokenizer, config, shuffle):
    dataset = BoolQDataset(
        df,
        tokenizer,
        config["max_len"],
        config["stride"],
    )
    loader = DataLoader(
        dataset,
        shuffle=shuffle,
        num_workers=4,
        batch_size=config["batch_size"],
        collate_fn=dataset.get_collate_fn(),
    )
    return dataset, loader


def train(train_df, valid_df, config, wandb):
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    _, train_loader = get_boolq_dataset(train_df, tokenizer, config, shuffle=True)
    _, valid_loader = get_boolq_dataset(valid_df, tokenizer, config, shuffle=False)

    print(f"Loading model: {config['model_path']}")
    model = BoolQModel(
        config["model_path"],
        config["head"],
        dropout=config["dropout"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        config["lr"],
        betas=config["betas"],
        eps=config["eps"],
        weight_decay=config["wd"],
    )
    scheduler = get_linear_warmup_power_decay_scheduler(
        optimizer, config["warmup_steps"], float("inf"), power=-0.5
    )
    scaler = amp.GradScaler()

    print(f"Training...")
    return train_loop(
        model,
        device,
        optimizer,
        scheduler,
        scaler,
        train_loader,
        valid_loader,
        config,
        wandb,
    )


def get_df(dfs, config):
    return dfs[config["train_df"]], dfs[config["valid_df"]]


def run(dfs, config, wandb):
    config = copy.deepcopy(config)
    wandb.config = config
    set_seed(config["seed"])
    set_environ()
    print(f"Config: {json.dumps(config, indent=4, sort_keys=True)}")
    train_df, valid_df = get_df(dfs, config)

    best = train(train_df, valid_df, config, wandb)
    wandb.run.summary["best_error"] = best
    print(f"Best: {best}")


class NoopWandB:
    def __init__(self):
        self.run = SimpleNamespace(summary={})
        self.config = {}

    def log(self, *_args, **_kwargs):
        pass
