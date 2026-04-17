from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import Sort
from util import get_car, read_license_plate
from database import init_db, insert_record

# 初始化数据库
init_db()

# 加载模型
# YOLOv8n 用于通用车辆检测（COCO预训练）
coco_model = YOLO('yolov8n.pt')

# 自定义车牌检测模型（基于 CCPD2020 中文车牌数据集 fine-tuning）
license_plate_detector = YOLO('license_plate_detector.pt')

# SORT 多目标追踪器
mot_tracker = Sort()

# 视频输入
cap = cv2.VideoCapture('./sample.mp4')

# COCO 数据集中车辆相关类别 ID
vehicles = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# 存储所有帧的检测结果
results = {}

frame_nmr = -1
ret = True

while ret:
    frame_nmr += 1
    ret, frame = cap.read()

    if not ret:
        break

    results[frame_nmr] = {}

    # 1. 检测车辆
    detections = coco_model(frame)[0]
    detections_ = []

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # 2. SORT 追踪
    track_ids = mot_tracker.update(np.asarray(detections_))

    # 3. 检测车牌
    license_plates = license_plate_detector(frame)[0]

    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # 匹配归属车辆
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        if car_id != -1:
            # 裁剪车牌区域
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

            # 灰度化 + 二值化，提升 OCR 识别效果
            license_plate_crop_gray = cv2.cvtColor(
                license_plate_crop, cv2.COLOR_BGR2GRAY
            )
            _, license_plate_crop_thresh = cv2.threshold(
                license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
            )

            # 4. OCR 识别（PaddleOCR）
            license_plate_text, license_plate_text_score = read_license_plate(
                license_plate_crop_thresh
            )

            if license_plate_text is not None:
                # 5. 写入数据库记录
                insert_record(
                    plate_text=license_plate_text,
                    confidence=license_plate_text_score,
                    frame_id=frame_nmr,
                    car_id=int(car_id),
                    bbox=[x1, y1, x2, y2]
                )

                results[frame_nmr][car_id] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': license_plate_text,
                        'bbox_score': score,
                        'text_score': license_plate_text_score
                    }
                }

# 导出结果到 CSV
import csv

with open('test.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'frame_nmr', 'car_id',
        'car_bbox', 'license_plate_bbox',
        'license_plate_bbox_score', 'license_number',
        'license_number_score'
    ])
    for frame_id, cars in results.items():
        for car_id, info in cars.items():
            writer.writerow([
                frame_id, car_id,
                info['car']['bbox'],
                info['license_plate']['bbox'],
                info['license_plate']['bbox_score'],
                info['license_plate']['text'],
                info['license_plate']['text_score']
            ])

print(f"检测完成，共处理 {frame_nmr + 1} 帧，结果已保存至 test.csv 和 records.db")