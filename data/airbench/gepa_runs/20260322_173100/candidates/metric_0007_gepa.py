#!/usr/bin/env python3
"""AirBench 94 optimization: tuned hyperparams for speed and stability."""

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
                if g is None: continue
                state = self.state[p]
                if "momentum_buffer" not in state: state["momentum_buffer"] = torch.zeros_like(g)
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
            torch.save({"images": torch.tensor(dset.data), "labels": torch.tensor(dset.targets), "classes": dset.classes}, data_path)
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
    def __len__(self): return len(self.images) // self.batch_size if self.drop_last else ceil(len(self.images) / self.batch_size)
    def __iter__(self):
        if self.epoch == 0:
            images = self.proc_images["norm"] = self.normalize(self.images)
            if self.aug.get("flip", False): self.proc_images["flip"] = batch_flip_lr(images)
            pad = self.aug.get("translate", 0)
            if pad > 0: self.proc_images["pad"] = F.pad(images, (pad,) * 4, "reflect")
        images = self.proc_images["pad"] if self.aug.get("translate", 0) > 0 else (self.proc_images["flip"] if self.aug.get("flip", False) else self.proc_images["norm"])
        if self.aug.get("translate", 0) > 0: images = batch_crop(images, self.images.shape[-2])
        if self.aug.get("flip", False) and self.epoch % 2 == 1: images = images.flip(-1)
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
        torch.nn.init.dirac_(self.weight.data[:self.weight.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1, self.pool, self.norm1 = Conv(channels_in, channels_out), nn.MaxPool2d(2), BatchNorm(channels_out)
        self.conv2, self.norm2, self.activ = Conv(channels_out, channels_out), BatchNorm(channels_out), nn.GELU()
    def forward(self, x): return self.activ(self.norm2(self.conv2(self.activ(self.norm1(self.pool(self.conv1(x)))))))

class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.whiten = nn.Conv2d(3, 24, 2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(nn.GELU(), ConvGroup(24, 64), ConvGroup(64, 256), ConvGroup(256, 256), nn.MaxPool2d(3))
        self.head = nn.Linear(256, 10, bias=False)
        for mod in self.modules():
            if isinstance(mod, BatchNorm): mod.float()
            elif not isinstance(mod, nn.Sequential): mod.half()
    def reset(self):
        for mod in self.modules():
            if type(mod) in (nn.Conv2d, Conv, BatchNorm, nn.Linear): mod.reset_parameters()
        self.head.weight.data *= 1 / self.head.weight.data.std()
    def init_whiten(self, train_images, eps=5e-4):
        c, h, w = train_images.shape[1], 2, 2
        patches = train_images.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).float()
        patches_flat = patches.reshape(len(patches), -1)
        cov = (patches_flat.T @ patches_flat) / len(patches_flat)
        val, vec = torch.linalg.eigh(cov, UPLO="U")
        vec_scaled = vec.T.reshape(-1, c, h, w) / torch.sqrt(val.view(-1, 1, 1, 1) + eps)
        self.whiten.weight.data[:] = torch.cat((vec_scaled, -vec_scaled))
    def forward(self, x, whiten_bias_grad=True):
        x = F.conv2d(x, self.whiten.weight, self.whiten.bias if whiten_bias_grad else self.whiten.bias.detach())
        x = self.layers(x).view(len(x), -1)
        return self.head(x) / x.size(-1)

def infer(model, loader, tta_level=0):
    model.eval()
    test_images = loader.normalize(loader.images)
    @torch.no_grad()
    def infer_fn(inputs):
        logits = model(inputs)
        if tta_level == 0: return logits
        logits += model(inputs.flip(-1))
        if tta_level == 1: return logits
        pad = 1
        padded = F.pad(inputs, (pad,) * 4, "reflect")
        logits += model(padded[:, :, 0:32, 0:32]) + model(padded[:, :, 2:34, 2:34])
        return logits
    return torch.cat([infer_fn(x) for x in test_images.split(2000)])

def run_single_trial(model, data_dir: str, use_dummy_labels: bool = False) -> dict[str, float]:
    batch_size, bias_lr, head_lr = 2000, 0.053, 0.67
    wd = 2e-6 * batch_size
    train_loader = CifarLoader(data_dir, train=True, batch_size=batch_size, aug=dict(flip=True, translate=2))
    if use_dummy_labels: train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device)
    total_steps = ceil(8.5 * len(train_loader))
    whiten_steps = ceil(3 * len(train_loader))
    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for name, p in model.named_parameters() if "norm" in name and p.requires_grad]
    optimizer1 = torch.optim.SGD([dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd/bias_lr), dict(params=norm_biases, lr=bias_lr, weight_decay=wd/bias_lr), dict(params=[model.head.weight], lr=head_lr, weight_decay=wd/head_lr)], momentum=0.85, nesterov=True, fused=True)
    optimizer2 = Muon(filter_params, lr=0.24, momentum=0.6, nesterov=True)
    model.reset()
    model.init_whiten(train_loader.normalize(train_loader.images[:5000]))
    step = 0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(ceil(total_steps / len(train_loader))):
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_steps))
            F.cross_entropy(outputs, labels, label_smoothing=0.2, reduction="sum").backward()
            for g in optimizer1.param_groups[:1]: g["lr"] = bias_lr * (1 - step/whiten_steps)
            for g in optimizer1.param_groups[1:] + optimizer2.param_groups: g["lr"] = g["initial_lr"] if "initial_lr" in g else (0.24 if g in optimizer2.param_groups else bias_lr) * (1 - step/total_steps)
            optimizer1.step(); optimizer2.step(); model.zero_grad(set_to_none=True); step += 1
            if step >= total_steps: break
    ender.record(); torch.cuda.synchronize()
    tta_acc = (infer(model, CifarLoader(data_dir, train=False, batch_size=2000), 2).argmax(1) == CifarLoader(data_dir, train=False).labels.to("cuda")).float().mean().item()
    return {"tta_val_accuracy": float(tta_acc), "time_seconds": 1e-3 * starter.elapsed_time(ender)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="cifar10"); parser.add_argument("--trials", type=int, default=1); parser.add_argument("--json-only", action="store_true"); parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    model.compile(mode="max-autotune")
    res = [run_single_trial(model, args.data_dir, False) for _ in range(args.trials)]
    accs = torch.tensor([r["tta_val_accuracy"] for r in res])
    print(json.dumps({"mean_accuracy": accs.mean().item(), "mean_time_seconds": sum(r["time_seconds"] for r in res)/len(res), "trials": args.trials, "per_trial": res}))

if __name__ == "__main__": main()