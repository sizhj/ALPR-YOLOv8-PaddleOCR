"""
models.py
数据库表结构定义 + MySQL 连接配置

依赖：
    pip install sqlalchemy pymysql
"""

import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String,
    Float, DateTime, Boolean, ForeignKey, Text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# ── 数据库连接 ────────────────────────────────────────────
# 从环境变量读取，本地开发可在 .env 里配置
DB_USER     = os.getenv("DB_USER",     "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "123456")
DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = os.getenv("DB_PORT",     "3306")
DB_NAME     = os.getenv("DB_NAME",     "parking_db")

DATABASE_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,       # 自动检测断连
    pool_recycle=3600,        # 连接每小时回收一次
    echo=False                # 生产环境关闭 SQL 日志
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── 依赖注入：获取数据库会话 ──────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── 表结构 ───────────────────────────────────────────────

class User(Base):
    """管理员账户表"""
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    username      = Column(String(50), unique=True, nullable=False, index=True)
    hashed_password = Column(String(128), nullable=False)
    is_active     = Column(Boolean, default=True)
    created_at    = Column(DateTime, default=datetime.now)


class ParkingLot(Base):
    """停车场信息表"""
    __tablename__ = "parking_lots"

    id             = Column(Integer, primary_key=True, index=True)
    name           = Column(String(100), nullable=False)
    total_spaces   = Column(Integer, nullable=False)        # 总车位数
    hourly_rate    = Column(Float, default=3.0)             # 每小时收费（元）
    free_minutes   = Column(Integer, default=15)            # 免费时长（分钟）
    created_at     = Column(DateTime, default=datetime.now)

    records = relationship("ParkingRecord", back_populates="lot")


class ParkingRecord(Base):
    """
    车辆进出记录表

    核心业务表，记录每辆车的进出场信息。
    status: 'parked'（在场）/ 'exited'（已离场）
    """
    __tablename__ = "parking_records"

    id            = Column(Integer, primary_key=True, index=True)
    plate_number  = Column(String(20), nullable=False, index=True)  # 车牌号
    lot_id        = Column(Integer, ForeignKey("parking_lots.id"), nullable=False)
    entry_time    = Column(DateTime, nullable=False, default=datetime.now)
    exit_time     = Column(DateTime, nullable=True)                  # 出场前为 NULL
    fee           = Column(Float, nullable=True)                     # 出场后计算
    status        = Column(String(10), default="parked", index=True) # parked / exited
    created_at    = Column(DateTime, default=datetime.now)

    lot = relationship("ParkingLot", back_populates="records")


def init_db():
    """创建所有表，并插入默认停车场数据"""
    Base.metadata.create_all(bind=engine)

    # 插入默认停车场（首次运行）
    db = SessionLocal()
    try:
        if db.query(ParkingLot).count() == 0:
            default_lot = ParkingLot(
                name="智慧停车场1号",
                total_spaces=100,
                hourly_rate=3.0,
                free_minutes=15
            )
            db.add(default_lot)
            db.commit()
            print("默认停车场已创建")
    finally:
        db.close()
