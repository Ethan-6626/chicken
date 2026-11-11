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

网络使用 torchvision 的 ImageNet 预训练 ConvNeXt-Tiny，分类头替换为 2 类输出。训练集使用轻量数据增强和类别加权交叉熵，验证集使用 Youden 指数选择阈值。

```bash
.venv/bin/python train.py \
  --data dataset_combined \
  --output runs/combined_8_1_1_convnext_tiny \
  --epochs 30 \
  --batch-size 8 \
  --device cuda \
  --workers 2 \
  --seed 20261016
```

如果显存不足，可将 `--batch-size` 改为 4；如果只做代码冒烟测试，可增加 `--weights none`。

## 独立测试

```bash
.venv/bin/python test.py \
  --data dataset_combined \
  --checkpoint runs/combined_8_1_1_convnext_tiny/best.pt \
  --output runs/combined_8_1_1_convnext_tiny/test \
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
| Accuracy | 96.23% |
| ROC-AUC | 1.000 |
| F1 | 0.952 |
| Sensitivity | 1.000 |
| Specificity | 0.939 |

详细说明见 [report.md](report.md)。

## 重要限制

本次最终数据集采用图片级随机划分，同一原始大图产生的裁片可能进入不同集合，因此结果可能受到相似图片泄漏影响。当前结果适合作为基线，不应直接解释为对新病例、新设备或新批次的泛化性能。

如果需要更严格的评估，应按原始大图分组划分，或将第二批完全作为独立外部测试集。

## Git 提交说明

`.gitignore` 会忽略虚拟环境、数据集、模型权重、训练输出、Python 缓存和外部下载数据。建议提交以下可复现文件：

- `process_data.py`
- `train.py`
- `test.py`
- `requirements.txt`
- `README.md`
- `report.md`
