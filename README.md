# 抗原凝集图片二分类实验

本项目使用第一批和第二批凝集裁片训练 ConvNeXt-Tiny 二分类模型：

- `normal`：假阳性、阴性
- `diseased`：弱阳性、阳性、强阳性

当前最终数据集是两批数据合并后随机打乱，按图片数量 **8:1:1** 划分。

## 目录结构

```text
antigen_binary_experiment/
├── process_data.py          # 第二批裁片、两批合并和 8:1:1 划分
├── train.py                 # ConvNeXt-Tiny 训练
├── test.py                  # 独立测试和指标计算
├── report.md                # 本次实验报告
├── web/                     # 前端演示界面
├── requirements.txt         # Python 依赖
├── dataset_combined/        # 本地生成的数据集（已加入 .gitignore）
└── runs/                    # 本地模型、日志和预测结果（已加入 .gitignore）
```

第一批和第二批原始数据位于工程目录外的 `../raw_data/`：

- `../raw_data/抗原图片/`：第一批五分类裁片，共 418 张；
- `../raw_data/凝集图片20260827/`：第二批原始整图和 `结果统计.xlsx`；
- `../raw_data/凝集图片20260827_裁片/`：第二批生成的 224×224 裁片，共 106 张。

## 环境安装

建议使用 Python 3.10–3.12。CUDA 训练需要安装与本机驱动匹配的 PyTorch。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 数据处理

`process_data.py` 已合并两部分数据处理逻辑，并保留 Excel 标签到中文细分类的映射：

```text
-    → 阴性   → normal
+    → 弱阳性 → diseased
++   → 阳性   → diseased
+++  → 强阳性 → diseased
```

### 一步完成第二批裁片和最终划分

```bash
cd /home/ke/shupj/antigen_binary_experiment

.venv/bin/python process_data.py \
  --mode all \
  --second-source "../raw_data/凝集图片20260827" \
  --labels "../raw_data/凝集图片20260827/结果统计.xlsx" \
  --second-output "../raw_data/凝集图片20260827_裁片" \
  --first-source "../raw_data/抗原图片" \
  --output dataset_combined \
  --seed 20261110 \
  --overwrite
```

### 只处理第二批原图

```bash
.venv/bin/python process_data.py \
  --mode second \
  --second-source "../raw_data/凝集图片20260827" \
  --labels "../raw_data/凝集图片20260827/结果统计.xlsx" \
  --second-output "../raw_data/凝集图片20260827_裁片" \
  --overwrite
```

### 只合并已有的两批裁片

```bash
.venv/bin/python process_data.py \
  --mode combine \
  --first-source "../raw_data/抗原图片" \
  --second-output "../raw_data/凝集图片20260827_裁片" \
  --output dataset_combined \
  --seed 20261110 \
  --overwrite
```

当前划分结果为：

| 划分 | 图片数 | normal | diseased |
|---|---:|---:|---:|
| train | 419 | 218 | 201 |
| val | 52 | 29 | 23 |
| test | 53 | 33 | 20 |
| 合计 | 524 | 280 | 244 |

`dataset_combined/manifest.csv` 保存每张图片的来源、批次、细分类、二分类标签和目标路径；`split_summary.json` 保存划分统计。

## 模型训练

网络使用 torchvision 的 ImageNet 预训练 ConvNeXt-Tiny，分类头替换为 2 类输出。最终实验使用 tuned 数据增强、类别加权交叉熵，并在验证集上按验证集准确率选择阈值。

```bash
.venv/bin/python train.py \
  --data dataset_combined \
  --output runs/tuned_seed_20261112 \
  --epochs 40 --batch-size 8 --lr 1e-4 --weight-decay 0.02 \
  --augmentation tuned --threshold-objective accuracy \
  --class-weight-power 0.75 --label-smoothing 0.02 \
  --device cuda --workers 2 --seed 20261112
```

如果显存不足，可将 `--batch-size` 改为 4；如果只做代码冒烟测试，可增加 `--weights none`。

### 调参实验

`train.py` 支持仅用验证集选择阈值，以及调节增强、类别权重和标签平滑。例如：

```bash
.venv/bin/python train.py \
  --data dataset_combined \
  --output runs/tuned_seed_20261112 \
  --epochs 40 --batch-size 8 --lr 1e-4 --weight-decay 0.02 \
  --augmentation tuned --threshold-objective accuracy \
  --class-weight-power 0.75 --label-smoothing 0.02 \
  --seed 20261112 --device cuda --workers 2
```

本次三随机种子调参结果记录在 [report.md](report.md) 的“调参复现实验”部分。

## 独立测试

```bash
.venv/bin/python test.py \
  --data dataset_combined \
  --checkpoint runs/tuned_seed_20261112/best.pt \
  --output runs/tuned_seed_20261112/test \
  --threshold 0.5758489370346069 \
  --device cuda \
  --batch-size 32 \
  --workers 2
```

测试输出包括：

- `test_metrics.json`：准确率、ROC-AUC、F1、灵敏度、特异度和混淆矩阵；
- `test_predictions.csv`：逐张图片的真实标签、预测概率和预测结果。

## 本次实验结果

当前测试集结果：

| 指标 | 结果 |
|---|---:|
| Accuracy | **98.11%** |
| ROC-AUC | **1.000** |
| F1 | **0.976** |
| Sensitivity | **1.000** |
| Specificity | **0.970** |

详细说明见 [report.md](report.md)。

## 前端演示界面

项目包含一个无需构建工具的图片预测 Web 界面，提供图片上传、后端状态检查、患病概率和分类结果展示：

```bash
python3 -m http.server 8080 --directory web
```

浏览器打开 <http://localhost:8080> 即可。前端会调用配置的真实后端接口；如果接口不可用，才会明确标注为演示推理。

## 后端模型接口

使用最终模型启动真实预测接口：

```bash
.venv/bin/python -m uvicorn backend:app --host 127.0.0.1 --port 8787
```

健康检查：`GET /api/health`；图片预测：`POST /api/predict`，上传字段名为 `file`。接口加载 `runs/tuned_seed_20261112/best.pt`，使用验证集确定的阈值 `0.575848937`。

## 从数据处理到 WebUI 部署

### 1. 处理两批数据

确认原始数据目录位于项目同级的 `../raw_data/` 后，执行：

```bash
.venv/bin/python process_data.py \
  --mode all \
  --second-source "../raw_data/凝集图片20260827" \
  --labels "../raw_data/凝集图片20260827/结果统计.xlsx" \
  --second-output "../raw_data/凝集图片20260827_裁片" \
  --first-source "../raw_data/抗原图片" \
  --output dataset_combined \
  --seed 20261110 \
  --overwrite
```

### 2. 训练最终模型

```bash
.venv/bin/python train.py \
  --data dataset_combined \
  --output runs/tuned_seed_20261112 \
  --epochs 40 --batch-size 8 --lr 1e-4 --weight-decay 0.02 \
  --augmentation tuned --threshold-objective accuracy \
  --class-weight-power 0.75 --label-smoothing 0.02 \
  --seed 20261112 --device cuda --workers 2
```

### 3. 启动真实后端和 WebUI

后端会加载 `runs/tuned_seed_20261112/best.pt`。先启动接口：

```bash
.venv/bin/python -m uvicorn backend:app --host 127.0.0.1 --port 8787
```

后端根路径同时提供 WebUI，接口检查：

```bash
curl http://127.0.0.1:8787/api/health
```

如果只在服务器本机使用，可访问 `http://127.0.0.1:8787`。如果通过 Tailscale Serve/Funnel 转发，访问对应的 HTTPS 域名；前端会自动检查 `/api/health`，只有真实接口不可用时才显示演示状态。

### 4. 部署 GitHub Pages 前端

仓库地址：<https://github.com/Ethan-6626/chicken>

仓库中的 `docs/` 已准备好静态页面。在 GitHub 仓库的 `Settings → Pages` 中选择 `main` 分支和 `/docs` 目录，然后访问：

```text
https://ethan-6626.github.io/chicken/
```

GitHub Pages 只能托管前端，真实模型仍需保持后端服务在线，并确保前端能访问后端 HTTPS 地址。


