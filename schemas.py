"""
schemas.py
请求 / 响应数据结构定义（Pydantic）

FastAPI 用这些 schema 自动做参数校验和文档生成。
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ── 认证相关 ─────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str = Field(..., example="admin")
    password: str = Field(..., example="123456")

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── 停车场相关 ────────────────────────────────────────────

class ParkingLotResponse(BaseModel):
    id:              int
    name:            str
    total_spaces:    int
    available_spaces: int          # 从 Redis 实时读取
    hourly_rate:     float
    free_minutes:    int

    class Config:
        from_attributes = True


# ── 进出场相关 ────────────────────────────────────────────

class EntryRequest(BaseModel):
    plate_number: str = Field(..., example="粤B12345", description="车牌号码")
    lot_id:       int = Field(default=1, description="停车场ID")

class EntryResponse(BaseModel):
    record_id:    int
    plate_number: str
    lot_name:     str
    entry_time:   datetime
    message:      str = "进场成功"

class ExitRequest(BaseModel):
    plate_number: str = Field(..., example="粤B12345")
    lot_id:       int = Field(default=1)

class ExitResponse(BaseModel):
    record_id:    int
    plate_number: str
    entry_time:   datetime
    exit_time:    datetime
    duration_minutes: int
    fee:          float
    message:      str = "出场成功"


# ── 查询相关 ─────────────────────────────────────────────

class ParkingRecordResponse(BaseModel):
    id:           int
    plate_number: str
    lot_id:       int
    entry_time:   datetime
    exit_time:    Optional[datetime]
    fee:          Optional[float]
    status:       str

    class Config:
        from_attributes = True

class RecordListResponse(BaseModel):
    total:   int
    page:    int
    size:    int
    records: list[ParkingRecordResponse]
