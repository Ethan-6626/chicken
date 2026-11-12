from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# 设置种子
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Dataloader 读取数据
def loaders(root: Path, batch_size: int, workers: int, augmentation: str):
    # 训练集处理
    if augmentation == "tuned":
        # 调整增强强度，增加对位置、方向和亮度变化的鲁棒性
        train_tf = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.70, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(0.18, 0.18, 0.12, 0.03),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        # 保留原实验中的轻量增强
        train_tf = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.80, 1.0)),
                transforms.RandomRotation(5),
                transforms.ColorJitter(0.12, 0.12, 0.08, 0.02),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    # 验证集处理
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # 创建数据集
    sets = {
        s: datasets.ImageFolder(root / s, train_tf if s == "train" else eval_tf)
        for s in ("train", "val", "test")
    }
    data_loaders = {}

    use_pin_memory = torch.cuda.is_available()
    for split_name, dataset in sets.items():
        should_shuffle = split_name == "train"

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            num_workers=workers,
            pin_memory=use_pin_memory,
        )

        data_loaders[split_name] = data_loader
    return sets, data_loaders


# 模型预测
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, probs, losses = [], [], []
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(criterion(logits, y).item() * len(y))

        # 类别 0 表示患病
        probs.extend(torch.softmax(logits, 1)[:, 0].cpu().numpy())
        ys.extend((y == 0).int().cpu().numpy())
    return np.asarray(ys), np.asarray(probs), sum(losses) / len(ys)


# 指标
def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y, pred),
        "roc_auc": roc_auc_score(y, p),
        "f1": f1_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "sensitivity": recall_score(y, pred),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def choose_threshold(y, p, objective):
    """仅使用验证集标签和概率选择分类阈值。"""
    if objective == "youden":
        fpr, tpr, thresholds = roc_curve(y, p)
        valid = np.isfinite(thresholds)
        scores = np.where(valid, tpr - fpr, -np.inf)
        return float(thresholds[np.argmax(scores)])
    candidates = np.unique(np.r_[0.0, p, 1.0])
    scores = []
    for candidate in candidates:
        pred = (p >= candidate).astype(int)
        if objective == "accuracy":
            score = accuracy_score(y, pred)
        else:
            score = balanced_accuracy_score(y, pred)
        scores.append(score)
    best = np.max(scores)
    tied = candidates[np.isclose(scores, best)]
    return float(tied[np.argmin(np.abs(tied - 0.5))])


def main():
    # 读取命令行参数
    ap = argparse.ArgumentParser(description="ConvNeXt-Tiny 抗原二分类")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("runs/convnext_tiny"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument(
        "--weights",
        choices=["imagenet", "none"],
        default="imagenet",
        help="默认使用报告中的 ImageNet 预训练；none 仅用于离线冒烟测试",
    )
    ap.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="0 表示按报告完整训练全部轮次",
    )
    ap.add_argument("--augmentation", choices=["baseline", "tuned"], default="baseline")
    ap.add_argument(
        "--threshold-objective",
        choices=["youden", "accuracy", "balanced_accuracy"],
        default="youden",
    )
    ap.add_argument("--class-weight-power", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # 种子
    seed_everything(args.seed)

    # 设备
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前环境不可用")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    # 加载数据
    sets, dls = loaders(args.data, args.batch_size, args.workers, args.augmentation)

    # 预训练权重
    pretrained = (
        models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        if args.weights == "imagenet"
        else None
    )

    # 模型
    model = models.convnext_tiny(weights=pretrained)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
    model.to(device)

    counts = np.bincount(sets["train"].targets, minlength=2)
    base_weights = len(sets["train"]) / (2 * counts)
    weights = torch.tensor(
        base_weights**args.class_weight_power, dtype=torch.float32, device=device
    )
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history, best_auc, patience = [], -1.0, 0

    # 训练
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in dls["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * len(y)
        scheduler.step()
        vy, vp, vloss = predict(model, dls["val"], device)
        auc = roc_auc_score(vy, vp)
        row = {
            "epoch": epoch,
            "train_loss": running / len(sets["train"]),
            "val_loss": vloss,
            "val_auc": auc,
        }
        history.append(row)
        print(row)
        if auc > best_auc:
            best_auc, patience = auc, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "class_to_idx": sets["train"].class_to_idx,
                    "args": vars(args),
                },
                args.output / "best.pt",
            )
        else:
            patience += 1
            if (
                args.early_stopping_patience > 0
                and patience >= args.early_stopping_patience
            ):
                break
    model.load_state_dict(
        torch.load(args.output / "best.pt", map_location=device, weights_only=False)[
            "model"
        ]
    )
    vy, vp, _ = predict(model, dls["val"], device)
    threshold = choose_threshold(vy, vp, args.threshold_objective)
    ty, tp, _ = predict(model, dls["test"], device)
    result = {
        "device": str(device),
        "best_val_auc": best_auc,
        "validation": metrics(vy, vp, threshold),
        "test": metrics(ty, tp, threshold),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=history[0])
        w.writeheader()
        w.writerows(history)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
