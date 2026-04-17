"""
app.py
基于 Gradio 的车牌识别 Web Demo

运行方式：
    python app.py

访问 http://localhost:7860 即可使用
功能：
    - Tab1：上传图片，实时检测并识别车牌
    - Tab2：按车牌号查询历史识别记录
    - Tab3：查看交通流量统计（各车牌出现次数）
"""

import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
from util import read_license_plate
from database import init_db, insert_record, query_by_plate, query_recent, get_stats

# 初始化
init_db()
license_plate_detector = YOLO('license_plate_detector.pt')
coco_model = YOLO('yolov8n.pt')

VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck


def detect_image(image):
    """
    对单张图片进行车牌检测 + OCR 识别，返回标注图和识别结果。
    """
    if image is None:
        return None, "请上传图片"

    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    output_frame = frame.copy()
    found_plates = []

    # 车牌检测
    lp_results = license_plate_detector(frame)[0]

    for lp in lp_results.boxes.data.tolist():
        x1, y1, x2, y2, score, _ = lp
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # 预处理
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)

        # OCR
        plate_text, confidence = read_license_plate(thresh)

        # 绘制边界框
        color = (0, 255, 0) if plate_text else (0, 0, 255)
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

        if plate_text:
            label = f'{plate_text} ({confidence:.2f})'
            cv2.putText(
                output_frame, label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )
            found_plates.append({'plate': plate_text, 'confidence': confidence})
            # 存入数据库
            insert_record(plate_text, confidence)

    output_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

    if found_plates:
        result_text = '\n'.join(
            [f"✅ {p['plate']}  置信度: {p['confidence']:.2%}"
             for p in found_plates]
        )
    else:
        result_text = "⚠️ 未检测到车牌"

    return output_rgb, result_text


def search_records(plate_input):
    """按车牌号查询历史记录"""
    if not plate_input.strip():
        records = query_recent(50)
    else:
        records = query_by_plate(plate_input.strip())

    if not records:
        return [["暂无记录", "", "", ""]]

    return [
        [r['plate_text'], f"{r['confidence']:.2%}" if r.get('confidence') else '-',
         r.get('frame_id', '-'), r['timestamp']]
        for r in records
    ]


def show_stats():
    """显示交通流量统计"""
    stats = get_stats()
    if not stats:
        return [["暂无数据", "", ""]]
    return [[s['plate_text'], s['count'], s['last_seen']] for s in stats]


# 构建 Gradio 界面
with gr.Blocks(title="中文车牌识别系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚗 中文车牌识别系统
    基于 **YOLOv8 + PaddleOCR** 的智能车牌检测与识别，支持中文车牌（含汉字省份缩写）
    """)

    with gr.Tab("📷 车牌检测"):
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(label="上传图片", type="numpy")
                detect_btn = gr.Button("开始识别", variant="primary")
            with gr.Column():
                img_output = gr.Image(label="检测结果")
                text_output = gr.Textbox(label="识别车牌", lines=4)

        detect_btn.click(
            fn=detect_image,
            inputs=img_input,
            outputs=[img_output, text_output]
        )

    with gr.Tab("🔍 查询记录"):
        gr.Markdown("输入车牌号查询历史记录（支持模糊搜索，留空显示最近50条）")
        with gr.Row():
            search_input = gr.Textbox(
                label="车牌号码", placeholder="如：沪A12345 或 沪A"
            )
            search_btn = gr.Button("查询", variant="primary")

        records_table = gr.Dataframe(
            headers=["车牌号", "置信度", "帧编号", "识别时间"],
            label="查询结果"
        )
        search_btn.click(
            fn=search_records,
            inputs=search_input,
            outputs=records_table
        )

    with gr.Tab("📊 流量统计"):
        gr.Markdown("统计各车牌累计出现次数，用于交通流量分析")
        refresh_btn = gr.Button("刷新统计", variant="secondary")
        stats_table = gr.Dataframe(
            headers=["车牌号", "出现次数", "最后识别时间"],
            label="流量统计"
        )
        refresh_btn.click(fn=show_stats, outputs=stats_table)
        demo.load(fn=show_stats, outputs=stats_table)


if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False)
