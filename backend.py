"""真实 ConvNeXt-Tiny 推理接口。

启动：
    .venv/bin/uvicorn backend:app --host 127.0.0.1 --port 8787
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms
from torch import nn


ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path(os.getenv("ANTIGEN_CHECKPOINT", ROOT / "runs/tuned_seed_20261112/best.pt"))
THRESHOLD = 0.5758489370346069
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Antigen AI Inference API", version="20261112")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

eval_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def load_model() -> nn.Module:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"checkpoint not found: {CHECKPOINT}")
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE).eval()
    return model


model = load_model()


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": "ConvNeXt-Tiny",
        "checkpoint": str(CHECKPOINT),
        "device": str(DEVICE),
        "threshold": THRESHOLD,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="请上传 JPG、PNG 或 WebP 图片")
    try:
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="图片不能超过10 MB")
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="无法读取图片文件") from exc
    tensor = eval_transform(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probability = float(torch.softmax(model(tensor), dim=1)[0, 0].item())
    predicted = "diseased" if probability >= THRESHOLD else "normal"
    return {
        "filename": file.filename,
        "diseased_probability": probability,
        "threshold": THRESHOLD,
        "predicted_label": predicted,
        "predicted_label_zh": "患病状态" if predicted == "diseased" else "正常状态",
        "model": "ConvNeXt-Tiny",
        "seed": 20261112,
        "device": str(DEVICE),
    }
