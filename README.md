# 智慧停车场管理系统

基于 **YOLOv8 + PaddleOCR + FastAPI + MySQL + Redis** 的智慧停车场后端系统。

集成中文车牌自动识别与完整停车场业务管理，包含车辆进出场、自动计费、车位实时管理、JWT 认证等核心功能。

在 [Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8](https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8) 基础上进行了以下改进：

- 🔄 **数据集替换**：使用 CCPD2020 中文车牌数据集重新 fine-tuning，适配中国车牌格式
- 🔤 **OCR 升级**：将 EasyOCR 替换为 PaddleOCR，中文字符识别准确率从 71.3% 提升至 89.6%
- 🌐 **RESTful API**：使用 FastAPI 封装推理服务，提供车牌检测、历史查询、流量统计接口
- 🅿️ **停车场管理**：新增完整业务后端，实现车辆进出场、自动计费、车位管理
- 🔐 **用户认证**：JWT 实现管理员登录与接口权限控制
- ⚡ **Redis 缓存**：原子操作缓存实时车位状态，解决并发超卖问题
- 🗄️ **双数据库**：SQLite 持久化识别记录，MySQL 存储业务数据

---

## 效果展示

| 指标 | 改进前（EasyOCR）| 改进后（PaddleOCR）|
|------|------|------|
| 车牌检测 mAP@0.5 | 0.891 | **0.923** |
| OCR 字符准确率（含汉字）| 71.3% | **89.6%** |
| 单帧推理速度 | 38ms | 41ms |

> 测试集：CCPD2020 ccpd_base，共 2000 张，GPU: RTX 3060

---

## 项目结构

```
.
├── api.py               # 主入口，FastAPI 应用 + 路由注册
├── models.py            # MySQL 数据库表结构（SQLAlchemy）
├── schemas.py           # 请求/响应数据格式（Pydantic）
├── auth.py              # JWT 用户认证模块
├── parking.py           # 停车场业务接口（进出场/计费/统计）
├── cache.py             # Redis 车位缓存模块
├── app.py               # Gradio Web Demo
├── main.py              # 视频检测主流程（多目标追踪）
├── util.py              # OCR 工具函数（PaddleOCR）
├── database.py          # SQLite 识别记录存储
├── convert_ccpd.py      # CCPD2020 数据集格式转换
├── visualize.py         # 结果可视化
├── add_missing_data.py  # 帧插值补全
├── requirements.txt     # 依赖列表
└── datasets/
    └── ccpd/
        ├── train/
        ├── val/
        └── data.yaml
```

---

## 快速开始

### 1. 安装依赖

```bash
git clone https://github.com/sizhj/ALPR-YOLOv8-PaddleOCR
cd ALPR-YOLOv8-PaddleOCR
pip install -r requirements.txt
```

### 2. 安装 SORT 追踪模块

```bash
git clone https://github.com/abewley/sort
```

### 3. 配置数据库

安装 MySQL 和 Redis 后，创建数据库：

```sql
CREATE DATABASE parking_db CHARACTER SET utf8mb4;
```

根据需要修改 `models.py` 顶部的连接配置（或设置环境变量）：

```python
DB_USER     = os.getenv("DB_USER",     "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "123456")
DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_NAME     = os.getenv("DB_NAME",     "parking_db")
```

### 4. 启动 API 服务

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

首次启动自动完成：
- 创建 MySQL 数据表
- 初始化默认停车场（100个车位）
- 创建默认管理员账号：`admin / admin123`
- 同步 Redis 车位缓存

访问 [http://localhost:8000/docs](http://localhost:8000/docs) 查看完整接口文档

### 5. 启动 Gradio Web Demo（可选）

```bash
python app.py
```

访问 [http://localhost:7860](http://localhost:7860)

### 6. 准备数据集（可选，如需重新训练）

```bash
python convert_ccpd.py --src_dir ./CCPD2020/ccpd_base --dst_dir ./datasets/ccpd
yolo train model=yolov8n.pt data=./datasets/ccpd/data.yaml epochs=50 imgsz=640
```

---

## API 接口文档

### 🔐 用户认证

| 接口 | 方法 | 说明 |
|------|------|------|
| `/auth/login` | POST | 管理员登录，返回 JWT Token |
| `/auth/me` | GET | 获取当前登录用户信息 |

登录后在请求 Header 中携带：`Authorization: Bearer <token>`

### 🚗 车牌识别

| 接口 | 方法 | 说明 |
|------|------|------|
| `/detect` | POST | 上传图片，识别车牌号码 |
| `/records` | GET | 查询历史识别记录（支持模糊搜索） |
| `/records/{plate}` | GET | 精确查询单个车牌记录 |
| `/stats` | GET | 车牌出现频次统计 |

### 🅿️ 停车场管理

| 接口 | 方法 | 说明 | 需要登录 |
|------|------|------|------|
| `/parking/entry` | POST | 车辆进场 | ✅ |
| `/parking/exit` | POST | 车辆出场（自动计费）| ✅ |
| `/parking/current` | GET | 查询当前在场车辆 | ✅ |
| `/parking/records` | GET | 进出记录分页查询 | ✅ |
| `/parking/lot/{id}` | GET | 停车场实时车位状态 | ❌ |
| `/parking/stats` | GET | 营收与流量统计 | ✅ |

---

## 核心设计说明

### OCR 模块替换（util.py）

原始项目使用 EasyOCR，对中文字符识别效果较差，改用 PaddleOCR：

```python
# 改动前
reader = easyocr.Reader(['en'])
result = reader.readtext(crop_img)

# 改动后
ocr_engine = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
result = ocr_engine.ocr(license_plate_crop, cls=True)
```

### Redis 并发控制（cache.py）

用 Redis 原子操作解决高并发场景下的车位超卖问题：

```python
# 车辆进场：原子扣减，天然线程安全
remaining = redis.decr(f"parking:lot:{lot_id}:available")
if remaining < 0:
    redis.incr(...)  # 回滚
    raise HTTPException("车位已满")
```

### 自动计费逻辑（parking.py）

```python
def calculate_fee(entry, exit_, hourly_rate, free_minutes):
    total_minutes = (exit_ - entry).total_seconds() / 60
    billable_minutes = max(0, total_minutes - free_minutes)
    billable_hours = math.ceil(billable_minutes / 60)
    return round(billable_hours * hourly_rate, 2)
```

---

## 数据库设计

| 表名 | 说明 |
|------|------|
| `users` | 管理员账户（用户名、密码哈希）|
| `parking_lots` | 停车场信息（总车位、收费标准）|
| `parking_records` | 进出记录（车牌、进出时间、费用）|

---

## 依赖环境

- Python 3.10
- PyTorch 2.0+、Ultralytics YOLOv8、PaddleOCR 2.7
- FastAPI 0.104、Uvicorn 0.24
- MySQL 8.0+、SQLAlchemy 2.0、PyMySQL
- Redis 7.0+
- Gradio 4.7、OpenCV 4.8

---

## 参考

- 原始项目：[Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8](https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8)
- SORT 追踪：[abewley/sort](https://github.com/abewley/sort)
- CCPD 数据集：[detectRecog/CCPD](https://github.com/detectRecog/CCPD)
- PaddleOCR：[PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

---

## License

MIT
