"""
api.py
主入口 - 整合车牌识别 + 停车场管理两大模块

启动方式：
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

接口文档：
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

from database import init_db, insert_record, query_by_plate, query_recent, get_stats as get_plate_stats
from util import read_license_plate
from models import init_db as init_parking_db
from auth import init_admin, router as auth_router
from parking import router as parking_router
from cache import init_lot_cache, get_available_spaces
from models import SessionLocal, ParkingLot

# ── 应用初始化 ───────────────────────────────────────────
app = FastAPI(
    title="智慧停车场管理系统",
    description="""
## 接口模块

### 🚗 车牌识别（AI推理服务）
- `POST /detect` - 上传图片，识别车牌
- `GET  /records` - 查询历史识别记录
- `GET  /stats`   - 车流量统计

### 🅿️ 停车场管理（业务后端）
- `POST /parking/entry`   - 车辆进场
- `POST /parking/exit`    - 车辆出场（自动计费）
- `GET  /parking/current` - 当前在场车辆
- `GET  /parking/records` - 进出记录分页查询
- `GET  /parking/stats`   - 营收统计

### 🔐 用户认证
- `POST /auth/login` - 管理员登录，获取 JWT Token
- `GET  /auth/me`    - 获取当前用户信息
    """,
    version="2.0.0"
)

# 注册子路由
app.include_router(auth_router)
app.include_router(parking_router)

# 模型加载
license_plate_detector = YOLO("license_plate_detector.pt")


@app.on_event("startup")
def startup():
    """服务启动时初始化数据库、管理员、Redis缓存"""
    # 车牌识别数据库（SQLite）
    init_db()

    # 停车场数据库（MySQL）
    init_parking_db()

    # 默认管理员
    init_admin()

    # 同步 Redis 车位缓存（从 MySQL 计算实际可用车位数）
    db = SessionLocal()
    try:
        lots = db.query(ParkingLot).all()
        for lot in lots:
            from models import ParkingRecord
            parked_count = db.query(ParkingRecord).filter(
                ParkingRecord.lot_id == lot.id,
                ParkingRecord.status == "parked"
            ).count()
            available = lot.total_spaces - parked_count
            init_lot_cache(lot.id, available)
            print(f"停车场 [{lot.name}] 初始化：可用车位 {available}/{lot.total_spaces}")
    finally:
        db.close()


# ── 车牌识别接口 ─────────────────────────────────────────

def bytes_to_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="图片格式不支持，请上传 jpg/png")
    return img


@app.get("/", summary="健康检查", tags=["系统"])
def health_check():
    return {"status": "ok", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


@app.post("/detect", summary="车牌检测与识别", tags=["车牌识别"])
async def detect(file: UploadFile = File(...)):
    """上传图片，返回检测到的所有车牌及置信度，结果自动写入数据库"""
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
                "bbox":       [x1, y1, x2, y2]
            })

    return {"count": len(plates), "plates": plates}


@app.get("/records", summary="查询识别记录", tags=["车牌识别"])
def get_records(
    plate: Optional[str] = Query(None, description="车牌号模糊查询"),
    limit: int = Query(50, ge=1, le=200)
):
    records = query_by_plate(plate) if plate else query_recent(limit)
    return {"count": len(records), "records": records}


@app.get("/stats", summary="车流量统计", tags=["车牌识别"])
def stats():
    return {"stats": get_plate_stats()}
