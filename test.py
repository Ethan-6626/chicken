from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def build_loader(data_dir: Path, batch_size: int, workers: int):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    expected = {"diseased": 0, "normal": 1}
    if dataset.class_to_idx != expected:
        raise RuntimeError(f"类别目录应为 {expected}，实际为 {dataset.class_to_idx}")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def build_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_classes = checkpoint.get("class_to_idx")
    if checkpoint_classes and checkpoint_classes != {"diseased": 0, "normal": 1}:
        raise RuntimeError(f"模型类别映射不匹配: {checkpoint_classes}")
    # 测试时不下载预训练权重，全部参数都从 best.pt 恢复。
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model


@torch.inference_mode()
def infer(model, dataset, loader, device):
    labels, probabilities = [], []
    offset = 0
    rows = []
    for images, targets in loader:
        logits = model(images.to(device))
        probs = torch.softmax(logits, dim=1)[:, 0].cpu().numpy()
        true_diseased = (targets == 0).int().numpy()
        labels.extend(true_diseased.tolist())
        probabilities.extend(probs.tolist())
        for index, (target, probability) in enumerate(zip(true_diseased, probs)):
            path, _ = dataset.samples[offset + index]
            rows.append({
                "path": str(Path(path).resolve()),
                "true_label": "diseased" if int(target) == 1 else "normal",
                "diseased_probability": float(probability),
            })
        offset += len(targets)
    return np.asarray(labels), np.asarray(probabilities), rows


def choose_threshold(y_true, probabilities):
    if len(np.unique(y_true)) < 2:
        raise RuntimeError("验证集必须同时包含 normal 和 diseased 才能自动选择阈值")
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    finite = np.isfinite(thresholds)
    index = np.argmax(np.where(finite, tpr - fpr, -np.inf))
    return float(thresholds[index])


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    result = {
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "normal_samples": int((y_true == 0).sum()),
        "diseased_samples": int((y_true == 1).sum()),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "sensitivity": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    result["roc_auc"] = float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None
    return result, predictions


def main():
    parser = argparse.ArgumentParser(description="评估 ConvNeXt-Tiny 抗原二分类模型")
    parser.add_argument("--data", type=Path, required=True, help="包含 val/ 和 test/ 的数据集目录")
    parser.add_argument("--checkpoint", type=Path, required=True, help="train.py 生成的 best.pt")
    parser.add_argument("--output", type=Path, default=Path("runs/convnext_tiny/test"))
    parser.add_argument("--threshold", type=float, default=None,
                        help="患病阈值；省略时根据验证集 Youden 指数自动选择")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前环境不可用")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    model = build_model(args.checkpoint, device)

    threshold = args.threshold
    threshold_source = "command_line"
    if threshold is None:
        val_set, val_loader = build_loader(args.data / "val", args.batch_size, args.workers)
        val_y, val_prob, _ = infer(model, val_set, val_loader, device)
        threshold = choose_threshold(val_y, val_prob)
        threshold_source = "validation_youden_index"
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("阈值必须在 0 到 1 之间")

    test_set, test_loader = build_loader(args.data / "test", args.batch_size, args.workers)
    y_true, probabilities, rows = infer(model, test_set, test_loader, device)
    metrics, predictions = calculate_metrics(y_true, probabilities, threshold)
    metrics.update({"device": str(device), "threshold_source": threshold_source,
                    "checkpoint": str(args.checkpoint.resolve())})

    for row, prediction in zip(rows, predictions):
        row["predicted_label"] = "diseased" if int(prediction) == 1 else "normal"
        row["correct"] = row["true_label"] == row["predicted_label"]
    with (args.output / "test_predictions.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
