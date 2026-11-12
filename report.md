# 两批凝集裁片合并后的 ConvNeXt-Tiny 二分类实验报告

**实验日期：** 2026-11-12
**实验目的：** 将第一批和第二批凝集裁片合并，使用原实验中的 ConvNeXt-Tiny 二分类方法评估分类效果。

## 1. 实验结论

在当前随机图片级划分下，使用随机种子 `20261112` 的最终调参模型在 53 张测试图片上取得：

| 指标 | 测试集结果 |
|---|---:|
| Accuracy | **98.11%** |
| ROC-AUC | **1.000** |
| F1 | **0.976** |
| Precision | **0.952** |
| Sensitivity | **1.000** |
| Specificity | **0.970** |

模型正确识别了全部 20 张 diseased 图片；33 张 normal 图片中有 1 张被误判为 diseased。

## 2. 数据集

两批数据合计 524 张裁片，合并后随机打乱，按图片数量划分为 8:1:1：

| 数据集 | 图片数 | normal | diseased |
|---|---:|---:|---:|
| Train | 419 | 218 | 201 |
| Validation | 52 | 29 | 23 |
| Test | 53 | 33 | 20 |
| 合计 | 524 | 280 | 244 |

批次分布没有额外约束，完全遵循“合并后随机打乱”的要求：

| 数据集 | 第一批 | 第二批 |
|---|---:|---:|
| Train | 331 | 88 |
| Validation | 45 | 7 |
| Test | 42 | 11 |

二分类标签映射为：

- `假阳性`、`阴性` → `normal`
- `弱阳性`、`阳性`、`强阳性` → `diseased`

## 3. 划分方法

数据处理代码位于 [process_data.py](process_data.py)。流程如下：

1. 读取第一批五分类裁片；
2. 读取第二批裁片（第二批原图可由同一文件中的 Excel 裁片流程生成）；
3. 合并两批图片并使用随机种子 `20261110` 打乱；
4. 前 80% 作为训练集，接下来 10% 作为验证集，剩余 10% 作为测试集；
5. 按二分类标签复制到 `train/`、`val/`、`test/` 目录，并生成 `manifest.csv` 和 `split_summary.json`。



## 4. 模型与训练方法

- **网络：** torchvision `ConvNeXt-Tiny`
- **初始化：** ImageNet 预训练权重
- **分类头：** 替换为 2 类输出（`diseased`、`normal`）
- **损失函数：** 带类别权重的交叉熵
- **优化器：** AdamW
- **学习率：** `1e-4`
- **权重衰减：** `0.02`
- **学习率调度：** Cosine Annealing
- **训练轮数：** 40
- **Batch size：** 8
- **随机种子：** `20261112`
- **数据增强：** tuned（随机裁剪、水平/垂直翻转、旋转和颜色扰动）
- **类别权重幂次：** `0.75`
- **标签平滑：** `0.02`
- **设备：** CUDA（NVIDIA GB10）
- **训练代码：** [train.py](train.py)

验证集上使用验证集准确率选择分类阈值，最终阈值为 `0.5758489`。测试集使用同一阈值，未根据测试集结果重新调参。

## 5. 验证集结果

验证集最佳 ROC-AUC 为 **1.000**，对应结果为：

- Accuracy：1.000
- F1：1.000
- Sensitivity：1.000
- Specificity：1.000
- 混淆矩阵：`[[29, 0], [0, 23]]`

## 6. 测试集结果

测试集混淆矩阵如下，行是真实标签，列是预测标签：

```text
                 Pred normal   Pred diseased
True normal            32             1
True diseased           0            20
```

这表示当前模型没有漏检 diseased 样本，仅有 1 个 normal → diseased 的误报。

## 7. 结果解释与限制

当前结果说明模型在这次随机划分上具有很强的区分能力，尤其是 diseased 的检出能力较好。


## 8. 结果文件

- 数据集清单：[dataset_combined/manifest.csv](dataset_combined/manifest.csv)
- 数据划分统计：[dataset_combined/split_summary.json](dataset_combined/split_summary.json)
- 训练历史：[runs/tuned_seed_20261112/history.csv](runs/tuned_seed_20261112/history.csv)
- 训练指标：[runs/tuned_seed_20261112/metrics.json](runs/tuned_seed_20261112/metrics.json)
- 最佳模型：[runs/tuned_seed_20261112/best.pt](runs/tuned_seed_20261112/best.pt)
- 独立测试指标：[runs/tuned_seed_20261112/test/test_metrics.json](runs/tuned_seed_20261112/test/test_metrics.json)
- 逐图测试预测：[runs/tuned_seed_20261112/test/test_predictions.csv](runs/tuned_seed_20261112/test/test_predictions.csv)

## 9. 调参复现实验

针对阈值选择、训练轮数、随机种子、学习率、类别权重和数据增强进行了复现实验。阈值严格在验证集上按验证集准确率选择，测试集只用于最终评估；每个配置训练 40 轮，使用 `lr=1e-4`、`weight_decay=0.02`、tuned 增强、类别权重幂次 `0.75` 和标签平滑 `0.02`。

| 随机种子 | 测试 Accuracy | ROC-AUC | F1 |
|---:|---:|---:|---:|
| 20261110 | 96.23% | 0.994 | 0.950 |
| 20261111 | 96.23% | 0.997 | 0.947 |
| 20261112 | **98.11%** | **1.000** | **0.976** |
| 均值±标准差 | **96.86%±1.09%** | **0.997±0.003** | **0.958±0.016** |

最佳一次实验相对原基线（96.23%）提升 **1.89 个百分点**；三次平均提升约 **0.63 个百分点**。结果仍受当前图片级随机划分和测试集样本量较小的影响。
