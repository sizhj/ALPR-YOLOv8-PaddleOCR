"""
database.py
车牌识别记录存储模块

新增功能：将每次识别结果持久化到本地 SQLite 数据库，
支持按车牌号查询历史记录，便于后续统计分析（如车流量统计）。
"""

import sqlite3
import json
from datetime import datetime

DB_PATH = 'records.db'


def init_db():
    """初始化数据库，创建记录表（如不存在）"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS plate_records (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_text  TEXT    NOT NULL,
            confidence  REAL    NOT NULL,
            frame_id    INTEGER,
            car_id      INTEGER,
            bbox        TEXT,
            timestamp   TEXT    NOT NULL
        )
    ''')
    # 创建索引，加速按车牌查询
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_plate_text
        ON plate_records(plate_text)
    ''')
    conn.commit()
    conn.close()


def insert_record(plate_text, confidence, frame_id=None, car_id=None, bbox=None):
    """
    插入一条识别记录。

    Args:
        plate_text  (str):   识别出的车牌号码
        confidence  (float): OCR 置信度 (0~1)
        frame_id    (int):   视频帧编号
        car_id      (int):   SORT 追踪 ID
        bbox        (list):  车牌边界框 [x1, y1, x2, y2]
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        '''INSERT INTO plate_records
           (plate_text, confidence, frame_id, car_id, bbox, timestamp)
           VALUES (?, ?, ?, ?, ?, ?)''',
        (
            plate_text,
            round(confidence, 4),
            frame_id,
            car_id,
            json.dumps(bbox) if bbox else None,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
    )
    conn.commit()
    conn.close()


def query_by_plate(plate_text):
    """
    按车牌号查询所有历史记录。

    Args:
        plate_text (str): 车牌号码（支持模糊查询，如 '沪A'）

    Returns:
        list[dict]: 记录列表
    """
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        '''SELECT id, plate_text, confidence, frame_id, car_id, bbox, timestamp
           FROM plate_records
           WHERE plate_text LIKE ?
           ORDER BY timestamp DESC''',
        (f'%{plate_text}%',)
    ).fetchall()
    conn.close()

    return [
        {
            'id': r[0],
            'plate_text': r[1],
            'confidence': r[2],
            'frame_id': r[3],
            'car_id': r[4],
            'bbox': json.loads(r[5]) if r[5] else None,
            'timestamp': r[6]
        }
        for r in rows
    ]


def query_recent(limit=50):
    """
    查询最近 N 条记录。

    Args:
        limit (int): 返回条数上限，默认 50

    Returns:
        list[dict]: 记录列表
    """
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        '''SELECT id, plate_text, confidence, frame_id, timestamp
           FROM plate_records
           ORDER BY id DESC
           LIMIT ?''',
        (limit,)
    ).fetchall()
    conn.close()

    return [
        {
            'id': r[0],
            'plate_text': r[1],
            'confidence': r[2],
            'frame_id': r[3],
            'timestamp': r[4]
        }
        for r in rows
    ]


def get_stats():
    """
    统计各车牌出现次数，用于交通流量分析。

    Returns:
        list[dict]: [{'plate_text': ..., 'count': ..., 'last_seen': ...}]
    """
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        '''SELECT plate_text,
                  COUNT(*) as cnt,
                  MAX(timestamp) as last_seen
           FROM plate_records
           GROUP BY plate_text
           ORDER BY cnt DESC
           LIMIT 100'''
    ).fetchall()
    conn.close()

    return [
        {'plate_text': r[0], 'count': r[1], 'last_seen': r[2]}
        for r in rows
    ]
