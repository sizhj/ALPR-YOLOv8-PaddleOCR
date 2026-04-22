"""
api.py
车牌识别 RESTful API 服务

启动方式：
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

接口文档（启动后访问）：
    http://localhost:8000/docs
"""

import io
import cv2
import numpy as np
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from database import init_db, insert_record, query_by_plate, query_recent, get_stats
from util import read_license_plate

# ── 初始化 ──────────────────────────────────────────────
app = FastAPI(
    title="车牌识别 API",
    description="基于 YOLOv8 + PaddleOCR 的中文车牌识别服务",
    version="1.0.0"
)

init_db()
license_plate_detector = YOLO("license_plate_detector.pt")


# ── 工具函数 ─────────────────────────────────────────────
def bytes_to_image(data: bytes) -> np.ndarray:
    """将上传的图片字节流转为 OpenCV 格式"""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="图片格式不支持，请上传 jpg/png")
    return img


# ── 接口 ─────────────────────────────────────────────────

@app.get("/", summary="健康检查")
def health_check():
    """确认服务正常运行"""
    return {"status": "ok", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


@app.post("/detect", summary="车牌检测与识别")
async def detect(file: UploadFile = File(..., description="上传车辆图片（jpg/png）")):
    """
    上传一张图片，返回检测到的所有车牌及识别结果。

    - 自动检测图中所有车牌位置
    - 对每个车牌进行 OCR 识别
    - 识别结果自动写入数据库
    """
    image = bytes_to_image(await file.read())
    results = license_plate_detector(image)[0]
    plates = []

    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, score, _ = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)

        plate_text, confidence = read_license_plate(thresh)

        if plate_text:
            insert_record(plate_text, confidence)
            plates.append({
                "plate":      plate_text,
                "confidence": round(confidence, 4),
                "bbox":       [x1, y1, x2, y2],
                "det_score":  round(score, 4)
            })

    return {
        "count":  len(plates),
        "plates": plates
    }


@app.get("/records", summary="查询识别记录")
def get_records(
    plate: Optional[str] = Query(None, description="车牌号（支持模糊查询，留空返回最近50条）"),
    limit: int = Query(50, ge=1, le=200, description="返回条数上限")
):
    """
    查询历史识别记录。

    - 传入 plate 参数时按车牌号模糊匹配
    - 不传时返回最近 N 条记录
    """
    if plate:
        records = query_by_plate(plate)
    else:
        records = query_recent(limit)

    return {
        "count":   len(records),
        "records": records
    }


@app.get("/records/{plate_text}", summary="精确查询单个车牌记录")
def get_record_by_plate(plate_text: str):
    """根据完整车牌号查询所有历史记录"""
    records = query_by_plate(plate_text)
    if not records:
        raise HTTPException(status_code=404, detail=f"未找到车牌 {plate_text} 的记录")
    return {
        "plate":   plate_text,
        "count":   len(records),
        "records": records
    }


@app.get("/stats", summary="交通流量统计")
def traffic_stats():
    """
    统计各车牌出现次数，按频次降序排列。
    可用于分析高频进出车辆、交通流量趋势。
    """
    stats = get_stats()
    return {
        "total_unique_plates": len(stats),
        "stats": stats
    }
