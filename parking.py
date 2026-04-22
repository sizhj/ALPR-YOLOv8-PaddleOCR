"""
parking.py
停车场核心业务接口

接口列表：
    POST /parking/entry          车辆进场
    POST /parking/exit           车辆出场（自动计费）
    GET  /parking/current        查询当前在场车辆
    GET  /parking/records        查询进出记录（分页）
    GET  /parking/lot/{lot_id}   查询停车场实时状态
    GET  /parking/stats          统计数据
"""

import math
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from models import ParkingRecord, ParkingLot, get_db
from schemas import (
    EntryRequest, EntryResponse,
    ExitRequest, ExitResponse,
    ParkingRecordResponse, RecordListResponse,
    ParkingLotResponse
)
from cache import get_available_spaces, decr_space, incr_space, init_lot_cache
from auth import get_current_user, User

router = APIRouter(prefix="/parking", tags=["停车场管理"])


# ── 计费逻辑 ─────────────────────────────────────────────

def calculate_fee(entry: datetime, exit_: datetime,
                  hourly_rate: float, free_minutes: int) -> float:
    """
    停车计费：
    - 免费时长内不收费
    - 超出部分按小时计费，不足1小时按1小时算

    Args:
        entry:       进场时间
        exit_:       出场时间
        hourly_rate: 每小时收费（元）
        free_minutes: 免费时长（分钟）

    Returns:
        应收费用（元），保留2位小数
    """
    total_minutes = (exit_ - entry).total_seconds() / 60
    billable_minutes = max(0, total_minutes - free_minutes)
    billable_hours = math.ceil(billable_minutes / 60)
    return round(billable_hours * hourly_rate, 2)


# ── 接口 ─────────────────────────────────────────────────

@router.post("/entry", response_model=EntryResponse, summary="车辆进场")
def vehicle_entry(
    req: EntryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    记录车辆进场。
    - 检查是否已有未出场记录（防重复进场）
    - 用 Redis 原子操作减少可用车位，防止并发超卖
    """
    # 检查车位是否已满
    available = get_available_spaces(req.lot_id)
    if available <= 0:
        raise HTTPException(status_code=400, detail="车位已满，无法进场")

    # 检查该车牌是否已在场
    existing = db.query(ParkingRecord).filter(
        ParkingRecord.plate_number == req.plate_number,
        ParkingRecord.lot_id == req.lot_id,
        ParkingRecord.status == "parked"
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"{req.plate_number} 已在场，请勿重复进场")

    # Redis 原子扣减车位
    remaining = decr_space(req.lot_id)
    if remaining < 0:
        raise HTTPException(status_code=400, detail="车位已满，无法进场")

    # 写入进场记录
    record = ParkingRecord(
        plate_number=req.plate_number,
        lot_id=req.lot_id,
        entry_time=datetime.now(),
        status="parked"
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    lot = db.query(ParkingLot).filter(ParkingLot.id == req.lot_id).first()
    return EntryResponse(
        record_id=record.id,
        plate_number=record.plate_number,
        lot_name=lot.name if lot else "",
        entry_time=record.entry_time
    )


@router.post("/exit", response_model=ExitResponse, summary="车辆出场")
def vehicle_exit(
    req: ExitRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    记录车辆出场，自动计算停车费用。
    - 查找最近一条在场记录
    - 根据停车时长和收费标准计算费用
    - 更新 Redis 车位缓存
    """
    record = db.query(ParkingRecord).filter(
        ParkingRecord.plate_number == req.plate_number,
        ParkingRecord.lot_id == req.lot_id,
        ParkingRecord.status == "parked"
    ).order_by(ParkingRecord.entry_time.desc()).first()

    if not record:
        raise HTTPException(status_code=404, detail=f"未找到 {req.plate_number} 的在场记录")

    lot = db.query(ParkingLot).filter(ParkingLot.id == req.lot_id).first()
    exit_time = datetime.now()
    fee = calculate_fee(
        record.entry_time, exit_time,
        lot.hourly_rate, lot.free_minutes
    )
    duration_minutes = int((exit_time - record.entry_time).total_seconds() / 60)

    # 更新记录
    record.exit_time = exit_time
    record.fee = fee
    record.status = "exited"
    db.commit()

    # Redis 归还车位
    incr_space(req.lot_id)

    return ExitResponse(
        record_id=record.id,
        plate_number=record.plate_number,
        entry_time=record.entry_time,
        exit_time=exit_time,
        duration_minutes=duration_minutes,
        fee=fee
    )


@router.get("/lot/{lot_id}", response_model=ParkingLotResponse, summary="停车场实时状态")
def get_lot_status(lot_id: int, db: Session = Depends(get_db)):
    """
    查询停车场当前状态，包括实时可用车位数（从 Redis 读取）。
    """
    lot = db.query(ParkingLot).filter(ParkingLot.id == lot_id).first()
    if not lot:
        raise HTTPException(status_code=404, detail="停车场不存在")

    available = get_available_spaces(lot_id)
    return ParkingLotResponse(
        id=lot.id,
        name=lot.name,
        total_spaces=lot.total_spaces,
        available_spaces=available,
        hourly_rate=lot.hourly_rate,
        free_minutes=lot.free_minutes
    )


@router.get("/current", summary="当前在场车辆")
def get_current_vehicles(
    lot_id: int = Query(1),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """查询当前停车场内所有在场车辆"""
    records = db.query(ParkingRecord).filter(
        ParkingRecord.lot_id == lot_id,
        ParkingRecord.status == "parked"
    ).order_by(ParkingRecord.entry_time.desc()).all()

    return {
        "lot_id": lot_id,
        "count": len(records),
        "vehicles": [
            {
                "plate_number": r.plate_number,
                "entry_time": r.entry_time,
                "duration_minutes": int(
                    (datetime.now() - r.entry_time).total_seconds() / 60
                )
            }
            for r in records
        ]
    }


@router.get("/records", response_model=RecordListResponse, summary="进出记录查询")
def get_records(
    lot_id:      int = Query(1),
    plate:       Optional[str] = Query(None, description="车牌号（模糊查询）"),
    status:      Optional[str] = Query(None, description="parked / exited"),
    page:        int = Query(1, ge=1),
    size:        int = Query(20, ge=1, le=100),
    db:          Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """分页查询进出记录，支持按车牌号和状态筛选"""
    query = db.query(ParkingRecord).filter(ParkingRecord.lot_id == lot_id)
    if plate:
        query = query.filter(ParkingRecord.plate_number.like(f"%{plate}%"))
    if status:
        query = query.filter(ParkingRecord.status == status)

    total = query.count()
    records = query.order_by(
        ParkingRecord.entry_time.desc()
    ).offset((page - 1) * size).limit(size).all()

    return RecordListResponse(
        total=total, page=page, size=size,
        records=records
    )


@router.get("/stats", summary="营收与流量统计")
def get_stats(
    lot_id: int = Query(1),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    统计停车场营收与车流量数据，可用于报表展示。
    """
    total_records = db.query(func.count(ParkingRecord.id)).filter(
        ParkingRecord.lot_id == lot_id
    ).scalar()

    total_revenue = db.query(func.sum(ParkingRecord.fee)).filter(
        ParkingRecord.lot_id == lot_id,
        ParkingRecord.status == "exited"
    ).scalar() or 0.0

    today = datetime.now().date()
    today_count = db.query(func.count(ParkingRecord.id)).filter(
        ParkingRecord.lot_id == lot_id,
        func.date(ParkingRecord.entry_time) == today
    ).scalar()

    today_revenue = db.query(func.sum(ParkingRecord.fee)).filter(
        ParkingRecord.lot_id == lot_id,
        func.date(ParkingRecord.entry_time) == today,
        ParkingRecord.status == "exited"
    ).scalar() or 0.0

    return {
        "lot_id":        lot_id,
        "total_records": total_records,
        "total_revenue": round(total_revenue, 2),
        "today_count":   today_count,
        "today_revenue": round(today_revenue, 2),
        "available_spaces": get_available_spaces(lot_id)
    }
