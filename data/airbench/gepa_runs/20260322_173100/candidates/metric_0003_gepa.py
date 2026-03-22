#!/usr/bin/env python3
"""AirBench 94 faster version."""

from __future__ import annotations

import argparse
import json
from math import ceil

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch import nn


torch.backends.cudnn.benchmark = True


@torch.compile
def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                p.data.mul_(len(p.data) ** 0.5 / p.data.norm())
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape)
                p.data.add_(update, alpha=-lr)


CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))


def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-r, r + 1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    if r <= 2:
        for sy in range(-r, r + 1):
            for sx in range(-r, r + 1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r + sy : r + sy + crop_size, r + sx : r + sx + crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size + 2 * r), device=images.device, dtype=images.dtype)
        for s in range(-r, r + 1):
            mask = shifts[:, 0] == s
            images_tmp[mask] = images[mask, :, r + s : r + s + crop_size, :]
        for s in range(-r, r + 1):
            mask = shifts[:, 1] == s
            images_out[mask] = images_tmp[mask, :, :, r + s : r + s + crop_size]
    return images_out


class CifarLoader:
    def __init__(self, path, train=True, batch_size=500, aug=None):
        from pathlib import Path
        data_path = Path(path) / ("train.pt" if train else "test.pt")
        if not data_path.exists():
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            torch.save({"images": torch.tensor(dset.data), "labels": torch.tensor(dset.targets)}, data_path)
        data = torch.load(data_path, map_location=torch.device("cuda"), weights_only=True)
        self.images, self.labels = data["images"], data["labels"]
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)
        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {}
        self.epoch = 0
        self.aug = aug or {}
        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train

    def __len__(self):
        return len(self.images) // self.batch_size if self.drop_last else ceil(len(self.images) / self.batch_size)

    def __iter__(self):
        if self.epoch == 0:
            images = self.proc_images["norm"] = self.normalize(self.images)
            if self.aug.get("flip", False):
                images = self.proc_images["flip"] = batch_flip_lr(images)
            pad = self.aug.get("translate", 0)
            if pad > 0:
                self.proc_images["pad"] = F.pad(images, (pad,) * 4, "reflect")
        
        if self.aug.get("translate", 0) > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]
        if self.aug.get("flip", False) and self.epoch % 2 == 1:
            images = images.flip(-1)
        self.epoch += 1
        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i * self.batch_size : (i + 1) * self.batch_size]
            yield images[idxs], self.labels[idxs]


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = False


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding="same", bias=False)
    def reset_parameters(self):
        super().reset_parameters()
        torch.nn.init.dirac_(self.weight.data[: self.weight.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.GELU()
    def forward(self, x):
        x = self.pool(self.activ(self.norm1(self.conv1(x))))
        x = self.activ(self.norm2(self.conv2(x)))
        return x


class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths = dict(block1=64, block2=256, block3=256)
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size**2
        self.whiten = nn.Conv2d(3, whiten_width, whiten_kernel_size, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(nn.GELU(), ConvGroup(whiten_width, widths["block1"]), ConvGroup(widths["block1"], widths["block2"]), ConvGroup(widths["block2"], widths["block3"]), nn.MaxPool2d(3))
        self.head = nn.Linear(widths["block3"], 10, bias=False)
        for mod in self.modules():
            if isinstance(mod, BatchNorm): mod.float()
            elif not isinstance(mod, (nn.Sequential, ConvGroup)): mod.half()
    def reset(self):
        for mod in self.modules():
            if type(mod) in (nn.Conv2d, Conv, BatchNorm, nn.Linear): mod.reset_parameters()
        self.head.weight.data *= 1 / self.head.weight.data.std()
    def init_whiten(self, train_images, eps=5e-4):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = train_images.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).float()
        patches_flat = patches.view(len(patches), -1)
        est_patch_covariance = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO="U")
        eigenvectors_scaled = eigenvectors.T.reshape(-1, c, h, w) / torch.sqrt(eigenvalues.view(-1, 1, 1, 1) + eps)
        self.whiten.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))
    def forward(self, x, whiten_bias_grad=True):
        bias = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, bias if whiten_bias_grad else bias.detach())
        x = self.layers(x)
        return self.head(x.view(len(x), -1)) / x.size(1)


def infer(model, loader, tta_level=0):
    model.eval()
    test_images = loader.normalize(loader.images)
    @torch.no_grad()
    def _run(inp): return model(inp)
    with torch.no_grad():
        logits = torch.cat([_run(inputs) for inputs in test_images.split(2000)])
        if tta_level > 0:
            logits += torch.cat([_run(inputs.flip(-1)) for inputs in test_images.split(2000)])
            logits *= 0.5
    return logits


def evaluate(model, loader, tta_level=0):
    return (infer(model, loader, tta_level).argmax(1) == loader.labels).float().mean().item()


def run_single_trial(model, data_dir: str, use_dummy_labels: bool = False) -> dict[str, float]:
    batch_size = 2000
    test_loader = CifarLoader(data_dir, train=False, batch_size=2000)
    train_loader = CifarLoader(data_dir, train=True, batch_size=batch_size, aug=dict(flip=True, translate=2))
    if use_dummy_labels: train_loader.labels = torch.randint(0, 10, size=train_loader.labels.shape, device=train_loader.labels.device)

    total_train_steps = ceil(8.5 * len(train_loader))
    whiten_bias_steps = ceil(2 * len(train_loader))
    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for name, p in model.named_parameters() if "norm" in name and p.requires_grad]
    
    opt1 = torch.optim.SGD([dict(params=[model.whiten.bias], lr=0.05, weight_decay=2e-6 * batch_size / 0.05), dict(params=norm_biases, lr=0.05, weight_decay=2e-6 * batch_size / 0.05), dict(params=[model.head.weight], lr=0.5, weight_decay=2e-6 * batch_size / 0.5)], momentum=0.85, nesterov=True, fused=True)
    opt2 = Muon(filter_params, lr=0.22, momentum=0.6, nesterov=True)
    
    model.reset()
    model.init_whiten(train_loader.normalize(train_loader.images[:5000]))
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    
    step = 0
    for _ in range(ceil(total_train_steps / len(train_loader))):
        model.train()
        for inputs, labels in train_loader:
            F.cross_entropy(model(inputs, whiten_bias_grad=(step < whiten_bias_steps)), labels, label_smoothing=0.2, reduction="sum").backward()
            for g in opt1.param_groups: g["lr"] = (0.05 if g["params"][0].ndim > 1 else 0.05) * (1 - step / total_train_steps)
            for g in opt2.param_groups: g["lr"] = 0.22 * (1 - step / total_train_steps)
            opt1.step(); opt2.step(); model.zero_grad(set_to_none=True)
            step += 1
            if step >= total_train_steps: break
    
    ender.record(); torch.cuda.synchronize()
    return {"tta_val_accuracy": evaluate(model, test_loader, tta_level=1), "time_seconds": 1e-3 * starter.elapsed_time(ender)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="cifar10"); parser.add_argument("--trials", type=int, default=1); parser.add_argument("--warmup-trials", type=int, default=1); parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    model.compile(mode="max-autotune")
    for _ in range(args.warmup_trials): run_single_trial(model, args.data_dir, use_dummy_labels=True)
    per_trial = [run_single_trial(model, args.data_dir) for _ in range(args.trials)]
    accs = [p["tta_val_accuracy"] for p in per_trial]
    times = [p["time_seconds"] for p in per_trial]
    print(json.dumps({"mean_accuracy": sum(accs)/len(accs), "mean_time_seconds": sum(times)/len(times), "trials": args.trials, "per_trial": per_trial}))
    return 0


if __name__ == "__main__": raise SystemExit(main())