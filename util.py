import string
import easyocr
from paddleocr import PaddleOCR

# 初始化 PaddleOCR，支持中文车牌识别
# 替换原有 EasyOCR，对中文字符（汉字省份缩写）识别效果更佳
ocr_engine = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)

# EasyOCR 保留作为备用引擎（英文/数字部分）
reader = easyocr.Reader(['en'], gpu=False)

def read_license_plate(license_plate_crop):
    """
    使用 PaddleOCR 识别车牌文字。
    对中文车牌（如 沪A·12345）比 EasyOCR 有更高的字符级准确率。

    Args:
        license_plate_crop: 裁剪后的车牌图像 (numpy array)

    Returns:
        tuple: (识别文字, 置信度) 或 (None, None)
    """
    result = ocr_engine.ocr(license_plate_crop, cls=True)

    if result is None or result[0] is None:
        return None, None

    detections = result[0]
    plate_text = ''
    total_score = 0.0
    count = 0

    for line in detections:
        text = line[1][0]
        score = line[1][1]
        # 过滤低置信度结果
        if score > 0.5:
            plate_text += text
            total_score += score
            count += 1

    if count == 0:
        return None, None

    avg_score = total_score / count
    # 清理无效字符，保留字母、数字、中文
    plate_text = ''.join(
        c for c in plate_text
        if c.isalnum() or '\u4e00' <= c <= '\u9fff'
    ).upper()

    if len(plate_text) < 4:
        return None, None

    return plate_text, avg_score


def get_car(license_plate, vehicle_track_ids):
    """
    根据车牌边界框，匹配对应的追踪车辆。

    Args:
        license_plate: 车牌检测结果 [x1, y1, x2, y2, score, class_id]
        vehicle_track_ids: 车辆追踪结果列表 [[x1,y1,x2,y2,track_id], ...]

    Returns:
        匹配车辆的 (x1, y1, x2, y2, track_id) 或 (-1,-1,-1,-1,-1)
    """
    x1, y1, x2, y2, score, class_id = license_plate

    for vehicle in vehicle_track_ids:
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return vehicle

    return -1, -1, -1, -1, -1