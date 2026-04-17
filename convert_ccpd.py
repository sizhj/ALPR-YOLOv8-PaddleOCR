"""
convert_ccpd.py
CCPD2020 数据集格式转换脚本

将 CCPD2020 中文车牌数据集转换为 YOLOv8 训练所需的格式。

CCPD 文件名编码规则：
  区域-亮度-模糊度-[x1&y1&x2&y2&x3&y3&x4&y4]-[tl_x&tl_y&br_x&br_y]-[label]-[序列].jpg
  其中第4段为四个角点坐标，第5段为 [左上x, 左上y, 右下x, 右下y]

使用方式：
    python convert_ccpd.py \
        --src_dir ./CCPD2020/ccpd_base \
        --dst_dir ./datasets/ccpd \
        --val_ratio 0.1

输出目录结构：
    datasets/ccpd/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── data.yaml
"""

import os
import shutil
import random
import argparse
from pathlib import Path


def parse_ccpd_bbox(filename):
    """
    从 CCPD 文件名解析车牌边界框。

    Returns:
        (x1, y1, x2, y2) 像素坐标，或 None（解析失败）
    """
    stem = Path(filename).stem
    parts = stem.split('-')
    if len(parts) < 5:
        return None

    try:
        bbox_part = parts[4]  # 第5段：左上&右下坐标
        coords = bbox_part.split('_')
        x1, y1 = map(int, coords[0].split('&'))
        x2, y2 = map(int, coords[1].split('&'))
        return x1, y1, x2, y2
    except (IndexError, ValueError):
        return None


def bbox_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """
    将像素坐标 (x1,y1,x2,y2) 转换为 YOLO 格式 (cx, cy, w, h)，归一化到 [0,1]。
    """
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return cx, cy, w, h


def convert(src_dir, dst_dir, val_ratio=0.1, img_w=720, img_h=1160):
    """
    主转换逻辑。

    Args:
        src_dir   (str): CCPD 原始图片目录
        dst_dir   (str): 输出目录
        val_ratio (float): 验证集比例
        img_w, img_h: CCPD 默认图片尺寸（720×1160）
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    all_images = list(src_path.glob('*.jpg'))
    if not all_images:
        print(f"[ERROR] 未找到图片：{src_dir}")
        return

    random.seed(42)
    random.shuffle(all_images)

    val_count = int(len(all_images) * val_ratio)
    val_images = set(str(p) for p in all_images[:val_count])
    train_images = all_images[val_count:]

    skipped = 0
    converted = 0

    for split, images in [('train', train_images), ('val', list(val_images))]:
        img_out = dst_path / split / 'images'
        lbl_out = dst_path / split / 'labels'
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        img_list = images if split == 'train' else [Path(p) for p in images]

        for img_path in img_list:
            img_path = Path(img_path)
            bbox = parse_ccpd_bbox(img_path.name)

            if bbox is None:
                skipped += 1
                continue

            x1, y1, x2, y2 = bbox
            cx, cy, w, h = bbox_to_yolo(x1, y1, x2, y2, img_w, img_h)

            # 边界检查
            if not (0 < cx < 1 and 0 < cy < 1 and 0 < w < 1 and 0 < h < 1):
                skipped += 1
                continue

            # 复制图片
            shutil.copy(img_path, img_out / img_path.name)

            # 写 YOLO label
            label_file = lbl_out / (img_path.stem + '.txt')
            with open(label_file, 'w') as f:
                f.write(f'0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')

            converted += 1

    # 生成 data.yaml
    yaml_path = dst_path / 'data.yaml'
    yaml_content = f"""# CCPD2020 中文车牌数据集
# 原始来源: https://github.com/detectRecog/CCPD
# 转换脚本: convert_ccpd.py

train: {(dst_path / 'train' / 'images').resolve()}
val:   {(dst_path / 'val'   / 'images').resolve()}

nc: 1
names: ['license_plate']
"""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"转换完成：{converted} 张成功，{skipped} 张跳过")
    print(f"data.yaml 已生成：{yaml_path}")
    print(f"\n训练命令：")
    print(f"  yolo train model=yolov8n.pt data={yaml_path} epochs=50 imgsz=640")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CCPD2020 → YOLOv8 格式转换')
    parser.add_argument('--src_dir',    required=True, help='CCPD 原始图片目录')
    parser.add_argument('--dst_dir',    default='./datasets/ccpd', help='输出目录')
    parser.add_argument('--val_ratio',  type=float, default=0.1, help='验证集比例')
    args = parser.parse_args()

    convert(args.src_dir, args.dst_dir, args.val_ratio)
