from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import zipfile
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image, ImageOps

# 第二批数据标签
SHEET_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
LABEL_TO_FOLDER = {
    "-": "阴性", 
    "+": "2弱阳性",
    "++": "3阳性",
    "+++": "4强阳性",
}

# 用于表格 Class 类
@dataclass(frozen=True)
class Grid:
    image: str
    rows: tuple[int, int]
    columns: tuple[str, str]
    x_max: int | None = None

# 表格中对应标签
GRIDS = (
    Grid("1.jpg", (3, 8), ("B", "G")),
    Grid("2.jpg", (11, 14), ("B", "G")),
    Grid("3.jpg", (17, 18), ("B", "F"), x_max=2600),
    Grid("4.jpg", (21, 24), ("B", "G")),
    Grid("5.jpg", (27, 28), ("B", "G")),
)

# 把 Excel 的列字母转换成数字列号
def column_number(name: str) -> int:
    result = 0
    for char in name:
        result = result * 26 + ord(char.upper()) - ord("A") + 1
    return result

# 读取 Excel 文件中的单元格
def read_sheet_labels(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = [
            "".join(node.text or "" for node in item.findall(".//m:t", SHEET_NS))
            for item in shared_root.findall("m:si", SHEET_NS)
        ]
        sheet_root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    values: dict[str, str] = {}
    for cell in sheet_root.findall(".//m:c", SHEET_NS):
        ref = cell.attrib["r"]
        node = cell.find("m:v", SHEET_NS)
        if node is None or node.text is None:
            continue
        values[ref] = shared[int(node.text)] if cell.attrib.get("t") == "s" else node.text
    return values

# 找第二批数据大图中样本的中心位置
def kmeans_1d(values: np.ndarray, clusters: int) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    centers = np.linspace(np.percentile(values, 2), np.percentile(values, 98), clusters)
    for _ in range(100):
        assignment = np.abs(values[:, None] - centers[None, :]).argmin(axis=1)
        updated = np.array([
            np.median(values[assignment == index])
            if np.any(assignment == index) else centers[index]
            for index in range(clusters)
        ])
        if np.max(np.abs(updated - centers)) < 0.1:
            break
        centers = updated
    return np.sort(centers)

# 判断像素是否为蓝色，生成蓝色掩码
def blue_mask(rgb: np.ndarray) -> np.ndarray:
    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
    return (blue - red > 25) & (blue - green > 15) & (blue > 100) & (red < 230)

# 计算分界线
def mid_boundaries(centers: np.ndarray, limit: int) -> np.ndarray:
    middle = (centers[:-1] + centers[1:]) / 2
    first = max(0.0, centers[0] - (centers[1] - centers[0]) / 2)
    last = min(float(limit), centers[-1] + (centers[-1] - centers[-2]) / 2)
    return np.concatenate(([first], middle, [last]))

# 裁剪样本转换为灰度图
def square_crop(image: Image.Image, mask: np.ndarray, bounds: tuple[int, int, int, int]) -> Image.Image:
    left, top, right, bottom = bounds
    local_y, local_x = np.nonzero(mask[top:bottom, left:right])
    if len(local_x) < 100:
        raise RuntimeError(f"裁剪单元中未检测到足够蓝色像素: {bounds}")

    x0, x1 = left + int(local_x.min()), left + int(local_x.max()) + 1
    y0, y1 = top + int(local_y.min()), top + int(local_y.max()) + 1
    width, height = x1 - x0, y1 - y0
    margin = max(10, round(max(width, height) * 0.10))
    side = max(width, height) + 2 * margin
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    crop_box = (
        round(center_x - side / 2),
        round(center_y - side / 2),
        round(center_x + side / 2),
        round(center_y + side / 2),
    )

    # PIL pads out-of-image areas with black, so paste onto an explicit white canvas.
    source = ImageOps.grayscale(image)
    canvas = Image.new("L", (side, side), 255)
    src_box = (
        max(0, crop_box[0]),
        max(0, crop_box[1]),
        min(image.width, crop_box[2]),
        min(image.height, crop_box[3]),
    )
    if src_box[2] <= src_box[0] or src_box[3] <= src_box[1]:
        raise RuntimeError(f"无效裁剪范围: {crop_box}")
    patch = source.crop(src_box)
    canvas.paste(patch, (src_box[0] - crop_box[0], src_box[1] - crop_box[1]))
    return canvas.resize((224, 224), Image.Resampling.LANCZOS).convert("RGB")


# 预处理第二批数据
def prepare(source: Path, labels_file: Path, output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"输出已存在: {output}；如需重建请加 --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    sheet = read_sheet_labels(labels_file)
    rows: list[dict[str, object]] = []

    for grid in GRIDS:
        image_path = source / grid.image
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        rgb = np.asarray(image)
        mask = blue_mask(rgb)
        if grid.x_max is not None:
            mask[:, grid.x_max:] = False

        row_count = grid.rows[1] - grid.rows[0] + 1
        column_count = column_number(grid.columns[1]) - column_number(grid.columns[0]) + 1
        y_pixels, x_pixels = np.nonzero(mask)
        y_centers = kmeans_1d(y_pixels, row_count)
        x_centers = kmeans_1d(x_pixels, column_count)
        y_bounds = mid_boundaries(y_centers, image.height)
        x_bounds = mid_boundaries(x_centers, image.width)

        sample_index = 0
        for row_index, sheet_row in enumerate(range(grid.rows[0], grid.rows[1] + 1)):
            for column_index, sheet_column in enumerate(
                range(column_number(grid.columns[0]), column_number(grid.columns[1]) + 1)
            ):
                column_name = chr(ord("A") + sheet_column - 1)
                cell = f"{column_name}{sheet_row}"
                label = sheet.get(cell)
                if label not in LABEL_TO_FOLDER:
                    raise RuntimeError(f"{grid.image} 对应单元格 {cell} 的标签无效: {label!r}")
                bounds = (
                    round(x_bounds[column_index]),
                    round(y_bounds[row_index]),
                    round(x_bounds[column_index + 1]),
                    round(y_bounds[row_index + 1]),
                )
                crop = square_crop(image, mask, bounds)
                name = f"batch_20260827_{Path(grid.image).stem}.jpg_img_{sample_index:05d}.jpg"
                destination = output / LABEL_TO_FOLDER[label] / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                crop.save(destination, quality=95)
                rows.append({
                    "source_image": grid.image,
                    "sheet_cell": cell,
                    "row": row_index + 1,
                    "column": column_index + 1,
                    "fine_symbol": label,
                    "fine_folder": LABEL_TO_FOLDER[label],
                    "filename": name,
                    "x_center": round(float(x_centers[column_index]), 2),
                    "y_center": round(float(y_centers[row_index]), 2),
                })
                sample_index += 1

    with (output / "crop_manifest.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"已生成 {len(rows)} 张裁片: {output}")


###### 第一批和第二批数据合并处理 ######

# 第一批和第二批数据的细分类标签映射为二分类标签
FINE_TO_BINARY = {
    "假阳性": "normal",
    "阴性": "normal",
    "弱阳性": "diseased",
    "阳性": "diseased",
    "强阳性": "diseased",
}

# 图片后缀
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# 获得图片所在目录对应的细分类标签
def fine_label(folder_name: str) -> str:
    name = re.sub(r"^\d+", "", folder_name)
    if name not in FINE_TO_BINARY:
        raise ValueError(f"无法识别类别目录: {folder_name}")
    return name


# 读取一批数据中的所有图片
def collect_images(source: Path, batch_name: str) -> list[dict[str, str]]:
    if not source.is_dir():
        raise FileNotFoundError(f"数据源目录不存在: {source}")

    records: list[dict[str, str]] = []
    for folder in sorted(path for path in source.iterdir() if path.is_dir()):
        fine = fine_label(folder.name)
        for image in sorted(folder.iterdir()):
            if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
                records.append({
                    "batch": batch_name,
                    "source_path": str(image.resolve()),
                    "source_folder": folder.name,
                    "original_filename": image.name,
                    "fine_label": fine,
                    "binary_label": FINE_TO_BINARY[fine],
                })
    return records


# 将两批图片合并、随机打乱，并按照 8:1:1 划分数据集
def combine_and_split(
    first_batch: Path,
    second_batch: Path,
    output: Path,
    seed: int,
    overwrite: bool,
) -> None:
    # 读取第一批和第二批图片
    records = collect_images(first_batch, "batch1")
    records.extend(collect_images(second_batch, "batch2"))
    if not records:
        raise RuntimeError("两批数据中没有找到图片")

    # 固定随机种子，保证每次可以复现相同的随机划分结果
    random.Random(seed).shuffle(records)

    # 按图片数量计算训练集、验证集和测试集的边界
    total = len(records)
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)
    split_ranges = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, total),
    }

    if output.exists():
        if not overwrite:
            raise FileExistsError(f"输出已存在: {output}；如需重建请加 --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    # 按二分类目录复制图片，并记录每张图片的来源和划分结果
    manifest: list[dict[str, str]] = []
    for split, (start, end) in split_ranges.items():
        for row in records[start:end]:
            source = Path(row["source_path"])
            # 加入批次和类别目录，避免两批数据出现同名文件时相互覆盖
            filename = (
                f"{row['batch']}__{row['source_folder']}__"
                f"{row['original_filename']}"
            )
            relative = Path(split) / row["binary_label"] / filename
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            manifest.append({
                **row,
                "split": split,
                "relative_path": relative.as_posix(),
            })

    # 保存合并数据集清单，便于追溯每张图片的原始位置
    with (output / "manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    # 保存各个划分的数量和类别统计
    summary = {
        "seed": seed,
        "method": "merge_then_random_shuffle_by_image",
        "ratios_requested": {"train": 0.8, "val": 0.1, "test": 0.1},
        "total_images": total,
        "counts": {},
    }
    for split in split_ranges:
        subset = [row for row in manifest if row["split"] == split]
        summary["counts"][split] = {
            "total": len(subset),
            "batch": dict(Counter(row["batch"] for row in subset)),
            "binary": dict(Counter(row["binary_label"] for row in subset)),
            "fine": dict(Counter(row["fine_label"] for row in subset)),
        }
    (output / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="处理第一批和第二批数据，并生成最终的 8:1:1 数据集"
    )
    parser.add_argument(
        "--mode", choices=("second", "combine", "all"), default="all",
        help="second: 只裁剪第二批；combine: 只合并划分；all: 先裁剪再合并",
    )
    parser.add_argument(
        "--second-source", type=Path,
        default=Path("../raw_data/凝集图片20260827"),
        help="第二批原始整图目录",
    )
    parser.add_argument(
        "--labels", type=Path,
        default=Path("../raw_data/凝集图片20260827/结果统计.xlsx"),
        help="第二批结果统计表格",
    )
    parser.add_argument(
        "--second-output", type=Path,
        default=Path("../raw_data/凝集图片20260827_裁片"),
        help="第二批裁片输出目录",
    )
    parser.add_argument(
        "--first-source", type=Path,
        default=Path("../raw_data/抗原图片"),
        help="第一批五分类裁片目录",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("dataset_combined"),
        help="最终合并数据集输出目录",
    )
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # 先将第二批整图裁剪成带标签的 224x224 图片
    if args.mode in ("second", "all"):
        prepare(args.second_source, args.labels, args.second_output, args.overwrite)

    # 再将第一批和第二批裁片合并，并随机划分为训练/验证/测试集
    if args.mode in ("combine", "all"):
        combine_and_split(
            args.first_source,
            args.second_output,
            args.output,
            args.seed,
            args.overwrite,
        )


if __name__ == "__main__":
    main()
