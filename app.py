"""
Edge AI Object Mass Prediction System - Interactive Web Dashboard.
Industrial Computer Vision & Edge AI Pipeline:
1. Data Preparation & Annotation (CVAT / Roboflow Format)
2. Model Architecture & Segmentation (YOLOv11n-seg / Adaptive Contour Engine)
3. 2D Metrology & Mass Estimation (Perspective Depth Geometry + AI MLP Hybrid)
4. Model Quantization (FP32 -> INT8 Post-Training Quantization PTQ)
5. Hardware Acceleration & Edge Inference (ONNX Runtime, OpenVINO, TensorRT)
"""
import os
# Prevent OpenBLAS thread allocation issues on Windows Python 3.14
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import json
import time
import gc
import subprocess
import sys

from config import (
    BASE_DIR, DATA_DIR, SAMPLES_DIR, MODELS_DIR, STATIC_DIR,
    OBJECT_DENSITIES, REFERENCE_MARKERS, DEFAULT_CAMERA_PARAMS, MODEL_CONFIGS
)
from core.perspective_geometry import PerspectiveGeometryEngine
from core.segmenter import ObjectSegmenter
from core.feature_extractor import FeatureExtractor
from core.ai_regressor import AIMassRegressor
from core.quantizer import ModelQuantizer
from core.edge_inference import EdgeInferenceEngine
from core.data_manager import DatasetManager
from core.industrial_pipeline import IndustrialAsyncPipeline, draw_industrial_hud

# Page configuration
st.set_page_config(
    page_title="Edge AI Metrology - Dự Đoán Khối Lượng",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
css_file = STATIC_DIR / "style.css"
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize Session State
if "data_manager" not in st.session_state:
    st.session_state.data_manager = DatasetManager()
if "ai_regressor" not in st.session_state:
    reg = AIMassRegressor(model_type="mlp")
    reg.load_from_checkpoint()
    st.session_state.ai_regressor = reg
if "segmenter" not in st.session_state:
    st.session_state.segmenter = ObjectSegmenter(use_yolo=True)
if "geometry_engine" not in st.session_state:
    st.session_state.geometry_engine = PerspectiveGeometryEngine(
        focal_length_px=DEFAULT_CAMERA_PARAMS["focal_length_px"],
        default_distance_z=DEFAULT_CAMERA_PARAMS["default_distance_z_mm"]
    )
if "edge_engine" not in st.session_state:
    st.session_state.edge_engine = EdgeInferenceEngine()

dm: DatasetManager = st.session_state.data_manager
regressor: AIMassRegressor = st.session_state.ai_regressor
segmenter: ObjectSegmenter = st.session_state.segmenter
geom_engine: PerspectiveGeometryEngine = st.session_state.geometry_engine
edge_engine: EdgeInferenceEngine = st.session_state.edge_engine

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.5rem;">
        <span style="font-size:1.8rem;">⚖️</span>
        <div>
            <div style="font-size:1.15rem; font-weight:800; color:#ffffff; letter-spacing:-0.01em;">EDGE AI METROLOGY</div>
            <div style="font-size:0.75rem; color:#06b6d4; font-weight:600; letter-spacing:0.05em;">PRECISION MASS PREDICTION</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="display:flex; gap:0.4rem; margin-bottom:0.8rem;">
        <span class="status-pill status-pill-emerald"><span class="pulse-dot"></span> Online</span>
        <span class="status-pill status-pill-cyan">{edge_engine.active_provider.replace('ExecutionProvider', '')}</span>
    </div>
    """, unsafe_allow_html=True)

    banner_img = STATIC_DIR / "metrology_banner.jpg"
    if banner_img.exists():
        st.image(str(banner_img), caption="Industrial Metrology Optical Scanner", use_container_width=True)
    st.divider()

    menu = st.radio(
        "Điều hướng hệ thống:",
        [
            "🎥 1. Dự đoán Trực tiếp (Camera / Ảnh)",
            "🏷️ 2. Giai đoạn 1: Chuẩn bị & Gán nhãn",
            "🔍 3. Giai đoạn 2: Phân đoạn (YOLOv11n-seg)",
            "📐 4. Giai đoạn 3: Đo lường & AI Hybrid",
            "⚡ 5. Giai đoạn 4: Lượng tử hóa (INT8)",
            "🚀 6. Giai đoạn 5: Tăng tốc Phần cứng"
        ]
    )

    st.divider()
    st.markdown("#### ⚙️ Cấu hình Đối tượng & Camera")
    
    fruit_icons = {
        "apple": "🍎 Táo (Apple)",
        "orange": "🍊 Cam (Orange)",
        "banana": "🍌 Chuối (Banana)",
        "mango": "🥭 Xoài (Mango)",
        "lemon": "🍋 Chanh (Lemon)",
        "tomato": "🍅 Cà chua (Tomato)",
        "custom": "⚙️ Vật thể tùy chỉnh"
    }
    
    selected_fruit = st.selectbox(
        "Loại vật thể đo lường:",
        options=list(OBJECT_DENSITIES.keys()),
        format_func=lambda k: fruit_icons.get(k, k),
        index=0
    )
    obj_spec = OBJECT_DENSITIES[selected_fruit]
    st.markdown(f"""
    <div class="telemetry-box" style="margin: 0.4rem 0;">
        <div class="telemetry-box-label">Mật độ chuẩn (g/cm³)</div>
        <div class="telemetry-box-val">{obj_spec['density']} <span style="font-size:0.75rem; color:#94a3b8;">g/cm³</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    calib_mode = st.radio(
        "Phương thức hiệu chuẩn tỷ lệ:",
        ["Phát hiện ArUco Marker (Tự động)", "Khoảng cách Camera cố định Z (mm)"]
    )
    
    marker_size_mm = 50.0
    manual_distance_z = 500.0
    focal_length_px = 800.0
    
    if calib_mode == "Phát hiện ArUco Marker (Tự động)":
        marker_size_mm = st.number_input("Kích thước ArUco (mm):", min_value=10.0, max_value=200.0, value=50.0, step=5.0)
    else:
        manual_distance_z = st.number_input("Khoảng cách Camera Z (mm):", min_value=100.0, max_value=2000.0, value=500.0, step=20.0)
        focal_length_px = st.number_input("Tiêu cự Camera f (pixel):", min_value=200.0, max_value=3000.0, value=800.0, step=50.0)
        geom_engine.calibrate_with_distance(manual_distance_z, focal_length_px)

    st.divider()
    st.markdown("#### 📥 Thước đo Chuẩn ArUco")
    aruco_p = DATA_DIR / "reference_markers" / "aruco_id0_50mm.png"
    if aruco_p.exists():
        with open(aruco_p, "rb") as f:
            st.download_button("📄 Tải Thước ArUco 50mm", f, file_name="aruco_50mm.png", mime="image/png", use_container_width=True)

# Helper function to process image with single-pass overlay and ONNX accelerated inference
def process_single_image(image_bgr, object_class, marker_mm, manual_z, manual_f):
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr, False, True, 1.0, []
        
    h_img, w_img = image_bgr.shape[:2]
    
    # 1. Detect ArUco Marker
    found_aruco, corners, ppm, aruco_meta = geom_engine.detect_aruco_marker(image_bgr, marker_real_size_mm=marker_mm)
    
    # 2. Fallback calibration
    needs_user_input_calib = False
    if not found_aruco:
        if manual_z and manual_f:
            ppm = geom_engine.calibrate_with_distance(manual_z, manual_f)
        else:
            needs_user_input_calib = True
            ppm = 1.0
            
    # 3. Object Segmentation
    detected_objs = segmenter.segment_image(image_bgr)
    
    annotated_img = image_bgr.copy()
    
    # Draw ArUco if found
    if found_aruco and corners is not None:
        c_pts = corners.astype(int)
        cv2.polylines(annotated_img, [c_pts], True, (0, 255, 0), 2)
        cv2.putText(annotated_img, f"ArUco {marker_mm}mm (PPM: {ppm:.2f})", 
                    (c_pts[0][0], max(20, c_pts[0][1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    
    # Unified overlay pass for all detected objects (single-pass blending avoids RAM spikes)
    if detected_objs:
        overlay = annotated_img.copy()
        for i, obj in enumerate(detected_objs):
            color = (180, 50, 220) if i % 2 == 0 else (50, 180, 220)
            mask = obj.get("mask")
            if mask is not None:
                overlay[mask > 0] = color
            elif "contour" in obj and obj["contour"] is not None:
                cv2.drawContours(overlay, [obj["contour"]], -1, color, -1)
        cv2.addWeighted(overlay, 0.35, annotated_img, 0.65, 0, annotated_img)
        del overlay
        
    results_list = []
    
    for i, obj in enumerate(detected_objs):
        cnt = obj["contour"]
        bbox = obj["bbox_xywh"]
        color = (180, 50, 220) if i % 2 == 0 else (50, 180, 220)
        
        # Draw contour & rotated rectangle
        if cnt is not None:
            cv2.drawContours(annotated_img, [cnt], -1, color, 2)
            
        # Calculate metric dimensions
        geom_res = geom_engine.compute_metric_dimensions(
            major_axis_px=obj["major_axis_px"],
            minor_axis_px=obj["minor_axis_px"],
            area_px=obj["area_px"],
            centroid=obj["centroid"],
            img_shape=image_bgr.shape,
            object_class=object_class
        )
        
        if geom_res["success"]:
            # Feature extraction
            feat_vec = FeatureExtractor.extract_features(obj, geom_res, image_bgr.shape)
            
            # Predict mass using accelerated ONNX Edge engine if available, with Scikit-learn fallback
            if edge_engine and edge_engine.session is not None:
                onnx_out = edge_engine.predict_mass(feat_vec)
                ai_mass_g = onnx_out.get("predicted_mass_g", regressor.predict(feat_vec))
            else:
                ai_mass_g = regressor.predict(feat_vec)
                
            geom_mass_g = geom_res["pure_geometric_mass_g"]
            
            # Annotate label
            label = f"{object_class.upper()}: AI {ai_mass_g:.1f}g (Geom {geom_mass_g:.1f}g)"
            x, y, w, h = bbox
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 1)
            cv2.putText(annotated_img, label, (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(annotated_img, label, (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
                        
            results_list.append({
                "index": i + 1,
                "morphology": obj,
                "geometry": geom_res,
                "feature_vector": feat_vec,
                "pure_geometric_mass_g": geom_mass_g,
                "ai_predicted_mass_g": ai_mass_g,
                "error_compensation_g": ai_mass_g - geom_mass_g
            })
            
    return annotated_img, found_aruco, needs_user_input_calib, ppm, results_list

# ==============================================================================
# TAB 1: DỰ ĐOÁN TRỰC TIẾP (LIVE CAMERA / SNAPSHOT / UPLOAD / BENCHMARK)
# ==============================================================================
if menu.startswith("🎥 1."):
    st.markdown("""
    <div class="top-hero-banner">
        <h1 class="hero-title">🎥 Hệ Thống Đo Lường & Dự Đoán Khối Lượng Thời Gian Thực</h1>
        <div class="hero-subtitle">
            Tích hợp phân đoạn mặt nạ (YOLOv11n-seg), hình học phối cảnh chiều sâu 3D (ArUco Optical Scale) 
            và mạng nơ-ron học sâu (MLP / INT8 ONNX Engine) để bù sai số phi tuyến tính.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    input_source = st.radio(
        "Nguồn ảnh đầu vào:", 
        [
            "🔴 Trực tiếp qua Camera / Webcam (Real-time Live Stream)",
            "📸 Chụp 1 ảnh từ Camera (Snapshot)", 
            "📁 Tải ảnh từ máy tính", 
            "🍎 Chọn ảnh mẫu Benchmark có sẵn"
        ], 
        horizontal=True
    )
    
    img_bgr = None
    sample_file_path = None
    
    # --------------------------------------------------------------------------
    # CHẾ ĐỘ 1: TRỰC TIẾP QUA CAMERA / WEBCAM (REAL-TIME LIVE STREAM)
    # --------------------------------------------------------------------------
    if input_source.startswith("🔴 Trực tiếp"):
        c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([2, 1, 2])
        with c_ctrl1:
            cam_idx = st.number_input("Cổng Camera (Camera Index):", min_value=0, max_value=5, value=0, step=1)
        with c_ctrl2:
            st.write("")
            st.write("")
            start_stream = st.checkbox("▶️ BẬT CAMERA", value=False, key="live_cam_toggle")
        with c_ctrl3:
            st.write("")
            st.write("")
            if st.button("🖥️ Mở Cửa Sổ Camera Riêng (60+ FPS)", use_container_width=True):
                subprocess.Popen([sys.executable, "run_camera.py", "--cam", str(cam_idx), "--class_name", selected_fruit])
                st.toast(f"Đã mở cửa sổ camera riêng với loại quả {selected_fruit.upper()}!", icon="🚀")

        if start_stream:
            stream_container = st.empty()
            telemetry_container = st.empty()
            
            pipeline = IndustrialAsyncPipeline(
                camera_index=cam_idx,
                initial_class=selected_fruit,
                width=1280,
                height=720
            )
            
            if not pipeline.start():
                st.error(f"Không thể kết nối với Camera {cam_idx}. Vui lòng kiểm tra cáp hoặc quyền truy cập camera.")
            else:
                try:
                    frame_count = 0
                    while start_stream:
                        frame, cur_results, aruco_info, calib_status, stream_fps, ai_fps, ai_latency_ms = pipeline.get_render_state()
                        if frame is None:
                            time.sleep(0.01)
                            continue
                            
                        frame_count += 1
                        pipeline.set_object_class(selected_fruit)
                        
                        # Render decoupled high-speed HUD overlay
                        ann_frame = draw_industrial_hud(
                            frame=frame.copy(),
                            results=cur_results,
                            aruco_info=aruco_info,
                            object_class=selected_fruit,
                            stream_fps=stream_fps,
                            ai_fps=ai_fps,
                            ai_latency_ms=ai_latency_ms,
                            calib_status=calib_status
                        )
                        
                        stream_container.image(cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        
                        if cur_results:
                            top_res = cur_results[0]
                            g = top_res["geometry"]
                            comp = top_res.get("error_compensation_g", 0.0)
                            sign = "+" if comp >= 0 else ""
                            fastest_delay_ms = 1000.0 / max(1.0, stream_fps)
                            
                            telemetry_container.markdown(f"""
                            <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 0.5rem;">
                                <div class="metrology-card metrology-card-emerald">
                                    <div class="card-label">🎯 Khối lượng Dự đoán AI</div>
                                    <div class="card-value-display">{top_res['ai_predicted_mass_g']:.1f}<span class="card-value-unit">g</span></div>
                                    <div class="card-footer-sub">Bù sai số: <b style="color:{'#34d399' if comp>=0 else '#f87171'}">{sign}{comp:.1f}g</b> so với hình học</div>
                                </div>
                                <div class="metrology-card">
                                    <div class="card-label">📐 Kích thước 3D & Thể tích</div>
                                    <div style="font-family:'JetBrains Mono'; font-size:1.15rem; font-weight:700; color:#ffffff; margin-top:0.3rem;">
                                        {g['length_mm']:.0f} × {g['width_mm']:.0f} × {g['height_mm']:.0f} <span style="font-size:0.85rem; color:#94a3b8;">mm</span>
                                    </div>
                                    <div class="card-footer-sub">Thể tích: <b style="color:#38bdf8;">{g['volume_cm3']:.1f} cm³</b></div>
                                </div>
                                <div class="metrology-card">
                                    <div class="card-label">⚡ Luồng Công Nghiệp (Zero Bottleneck)</div>
                                    <div class="card-value-display" style="color:#38bdf8;">{stream_fps:.1f}<span class="card-value-unit">FPS</span></div>
                                    <div class="card-footer-sub">AI Worker: <b style="color:#e2e8f0;">{ai_fps:.1f} FPS ({ai_latency_ms:.1f}ms)</b> | Độ trễ thấp nhất: <b style="color:#34d399;">{fastest_delay_ms:.1f}ms</b></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        # Smooth refresh rate for web socket push
                        time.sleep(0.03)
                        if frame_count % 90 == 0:
                            gc.collect()
                finally:
                    pipeline.stop()
                    gc.collect()
                    st.info("Đã đóng luồng Industrial Pipeline an toàn.")
                    
        st.markdown("---")
        
    elif input_source.startswith("📸 Chụp 1 ảnh"):
        cam_shot = st.camera_input("Chụp ảnh vật thể trên mặt bàn (đặt kèm ArUco hoặc thẻ chuẩn):")
        if cam_shot is not None:
            pil_img = Image.open(cam_shot)
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
    elif input_source.startswith("📁 Tải ảnh"):
        up_file = st.file_uploader("Chọn ảnh vật thể (JPG, PNG):", type=["jpg", "jpeg", "png"])
        if up_file is not None:
            pil_img = Image.open(up_file)
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
    elif input_source.startswith("🍎 Chọn ảnh mẫu"):
        sample_files = list(SAMPLES_DIR.glob("*.jpg"))
        if not sample_files:
            st.info("Chưa có ảnh mẫu benchmark trong thư mục data/samples. Bấm nút bên dưới để tạo:")
            if st.button("✨ Khởi tạo 20 mẫu Benchmark chuẩn có ArUco"):
                dm.generate_synthetic_benchmark_dataset(20)
                st.rerun()
        else:
            selected_sample = st.selectbox("Chọn ảnh mẫu benchmark:", [f.name for f in sample_files])
            sample_file_path = SAMPLES_DIR / selected_sample
            img_bgr = cv2.imread(str(sample_file_path))

    if img_bgr is not None:
        # Run Complete Inference
        annotated_bgr, found_marker, needs_calib, cur_ppm, res_list = process_single_image(
            img_bgr, selected_fruit, marker_size_mm, manual_distance_z, focal_length_px
        )
        
        # --- MISSING DATA PROMPT: CALIBRATION ---
        if needs_calib:
            st.markdown("""
            <div class="industrial-alert">
                <div class="alert-header">⚠️ YÊU CẦU DỮ LIỆU: Thiếu Thước Đo Hiệu Chuẩn (Calibration Scale)</div>
                <div class="alert-content">
                    Hệ thống không phát hiện thấy <b>ArUco Marker</b> trong khung hình. 
                    Để đo chính xác kích thước thực tế (mm) và thể tích/khối lượng, xin vui lòng cung cấp thông số bên dưới:
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            c_input1, c_input2, c_btn = st.columns([2, 2, 1])
            with c_input1:
                input_z = st.number_input("Khoảng cách Camera đến bàn Z (mm):", value=500.0, step=20.0, key="prompt_z")
            with c_input2:
                input_f = st.number_input("Tiêu cự Camera f (pixel):", value=800.0, step=50.0, key="prompt_f")
            with c_btn:
                st.write("")
                st.write("")
                if st.button("Áp dụng thông số"):
                    geom_engine.calibrate_with_distance(input_z, input_f)
                    st.rerun()

        # Display images and analysis side-by-side
        col_img, col_metrics = st.columns([3, 2])
        
        with col_img:
            st.markdown("##### 🖼️ Hình ảnh phân đoạn & Trực quan hóa phối cảnh")
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True)
            
            # Calibration Status Tag
            if found_marker:
                st.markdown(f"<span class='status-pill status-pill-emerald'>✓ Phát hiện ArUco Marker ({marker_size_mm}mm) | PPM: {cur_ppm:.2f} px/mm</span>", unsafe_allow_html=True)
            elif not needs_calib:
                st.markdown(f"<span class='status-pill status-pill-cyan'>✓ Hiệu chuẩn theo chiều sâu Z = {manual_distance_z}mm | PPM: {cur_ppm:.2f} px/mm</span>", unsafe_allow_html=True)

        with col_metrics:
            st.markdown("##### 📊 Kết quả Đo lường & Dự đoán Khối lượng")
            
            if not res_list:
                st.warning("Không tìm thấy vật thể nào đáp ứng tiêu chuẩn phân đoạn. Hãy căn chỉnh ánh sáng hoặc chọn vị trí rõ ràng hơn.")
            else:
                for item in res_list:
                    geom = item["geometry"]
                    ai_mass = item["ai_predicted_mass_g"]
                    geom_mass = item["pure_geometric_mass_g"]
                    comp = item["error_compensation_g"]
                    sign = "+" if comp >= 0 else ""
                    
                    st.markdown(f"""
                    <div class="metrology-card metrology-card-emerald">
                        <div class="card-label">🎯 Khối lượng Dự đoán AI Hybrid (MLP Model)</div>
                        <div class="card-value-display">{ai_mass:.1f}<span class="card-value-unit">gam</span></div>
                        <div class="card-footer-sub">
                            Bù sai số phối cảnh: <span class="comp-delta-tag {'delta-positive' if comp>=0 else 'delta-negative'}">{sign}{comp:.1f}g</span> so với hình học thuần
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"""
                        <div class="metrology-card">
                            <div class="card-label">📐 Kích thước 3D</div>
                            <div style="font-family:'JetBrains Mono'; font-size:1.1rem; font-weight:700; color:#ffffff;">
                                {geom['length_mm']:.1f} × {geom['width_mm']:.1f} × {geom['height_mm']:.1f} <span style="font-size:0.8rem; color:#94a3b8;">mm</span>
                            </div>
                            <div class="card-footer-sub">Thể tích: <b style="color:#38bdf8;">{geom['volume_cm3']:.1f} cm³</b></div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""
                        <div class="metrology-card">
                            <div class="card-label">⚖️ Khối lượng Hình học</div>
                            <div style="font-family:'JetBrains Mono'; font-size:1.1rem; font-weight:700; color:#cbd5e1;">
                                {geom_mass:.1f} <span style="font-size:0.8rem; color:#94a3b8;">gam</span>
                            </div>
                            <div class="card-footer-sub">Mật độ: {geom['density']} g/cm³</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # --- MISSING DATA PROMPT: GROUND TRUTH MASS ---
                    st.markdown("---")
                    st.markdown("##### 📝 Gán nhãn Khối lượng Thực tế (Ground Truth) để Tự động Học")
                    st.caption("Cân vật thể này bằng cân điện tử và nhập số gam thực tế để hệ thống đưa vào tập dữ liệu huấn luyện MLP:")
                    
                    gt_col1, gt_col2 = st.columns([2, 1])
                    with gt_col1:
                        user_gt_mass = st.number_input("Khối lượng cân thực tế (gram):", min_value=1.0, max_value=5000.0, value=float(round(ai_mass, 1)), key=f"gt_{item['index']}")
                    with gt_col2:
                        st.write("")
                        st.write("")
                        if st.button("💾 Lưu mẫu vào Dataset", key=f"btn_save_{item['index']}"):
                            save_path = sample_file_path or (SAMPLES_DIR / f"capture_{int(time.time())}.jpg")
                            if not Path(save_path).exists():
                                cv2.imwrite(str(save_path), img_bgr)
                                
                            res_save = dm.add_sample(
                                image_path=str(save_path),
                                object_class=selected_fruit,
                                feature_vector=item["feature_vector"],
                                geom_info=geom,
                                actual_weight_g=user_gt_mass
                            )
                            st.success(res_save["alert"])

# ==============================================================================
# TAB 2: GIAI ĐOẠN 1 - CHUẨN BỊ & GÁN NHÃN DỮ LIỆU
# ==============================================================================
elif menu.startswith("🏷️ 2."):
    st.markdown("""
    <div class="top-hero-banner">
        <h1 class="hero-title">🏷️ Giai đoạn 1: Chuẩn bị & Gán nhãn Dữ liệu (CVAT / Roboflow Format)</h1>
        <div class="hero-subtitle">
            Quản lý tập dữ liệu mẫu, đa giác phân đoạn (Segmentation Polygons), thông số phối cảnh và nhãn Ground Truth.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st_info = dm.check_dataset_status()
    
    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        st.markdown(f"""
        <div class="metrology-card">
            <div class="card-label">Tổng số mẫu trong Registry</div>
            <div class="card-value-display">{st_info['total_samples']}</div>
            <div class="card-footer-sub">Đã lưu trữ trong data/dataset.json</div>
        </div>
        """, unsafe_allow_html=True)
    with c_m2:
        st.markdown(f"""
        <div class="metrology-card metrology-card-emerald">
            <div class="card-label">Mẫu đã có Khối lượng chuẩn</div>
            <div class="card-value-display" style="color:#4ade80;">{st_info['labeled_samples']}</div>
            <div class="card-footer-sub">Đủ điều kiện huấn luyện AI MLP</div>
        </div>
        """, unsafe_allow_html=True)
    with c_m3:
        st.markdown(f"""
        <div class="metrology-card">
            <div class="card-label">Mẫu đang thiếu Khối lượng</div>
            <div class="card-value-display" style="color:#f87171;">{st_info['unlabeled_samples']}</div>
            <div class="card-footer-sub">Đang cần người dùng cân và nhập</div>
        </div>
        """, unsafe_allow_html=True)

    if st_info["alerts"]:
        for alert in st_info["alerts"]:
            st.markdown(f"""
            <div class="industrial-alert">
                <div class="alert-header">⚠️ THÔNG BÁO TẬP DỮ LIỆU</div>
                <div class="alert-content">{alert}</div>
            </div>
            """, unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("✨ Khởi tạo / Bổ sung 20 mẫu mô phỏng Benchmark chuẩn", use_container_width=True):
            n = dm.generate_synthetic_benchmark_dataset(20)
            st.success(f"Đã tạo thành công {n} mẫu trái cây đa dạng kèm ArUco marker!")
            st.rerun()
    with col_btn2:
        if st.button("🗑️ Xóa toàn bộ tập dữ liệu để làm lại từ đầu", use_container_width=True):
            dm.dataset = []
            dm.save_dataset()
            st.info("Đã xóa sạch bộ dữ liệu registry.")
            st.rerun()

    st.markdown("---")
    st.markdown("##### 📋 Bảng Dữ Liệu Chi Tiết Các Mẫu (Ground Truth Registry)")
    
    if len(dm.dataset) > 0:
        table_rows = []
        for d in dm.dataset:
            table_rows.append({
                "Mã mẫu": d["id"],
                "Loại vật thể": d["object_class"],
                "Khối lượng thực tế (g)": d.get("actual_weight_g"),
                "Khối lượng hình học (g)": round(d.get("pure_geometric_mass_g", 0), 1),
                "Dài (mm)": round(d.get("length_mm", 0), 1),
                "Rộng (mm)": round(d.get("width_mm", 0), 1),
                "Thể tích (cm³)": round(d.get("volume_cm3", 0), 1),
                "PPM (px/mm)": round(d.get("pixels_per_mm", 0), 2) if d.get("pixels_per_mm") else "N/A",
                "Trạng thái": "✅ Đầy đủ" if d.get("actual_weight_g") is not None else "⚠️ Thiếu cân nặng"
            })
        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Chưa có mẫu nào trong cơ sở dữ liệu.")

# ==============================================================================
# TAB 3: GIAI ĐOẠN 2 - KIẾN TRÚC MÔ HÌNH NHẬN DIỆN & PHÂN ĐOẠN ẢNH
# ==============================================================================
elif menu.startswith("🔍 3."):
    st.markdown("""
    <div class="top-hero-banner">
        <h1 class="hero-title">🔍 Giai đoạn 2: Kiến trúc Mô hình Nhận diện & Phân đoạn Ảnh</h1>
        <div class="hero-subtitle">
            Ứng dụng YOLOv11n-seg (Nano) hoặc MobileNetV4 Backbone + DeepLabV3+ / Adaptive Contour Decoder để bóc tách mặt nạ pixel.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stage-pipeline-bar">
        <div>
            <div class="pipeline-stage-title">Trạng thái Bộ Phân đoạn Hiện tại</div>
            <div style="color:#a5b4fc; font-size:0.88rem; margin-top:0.25rem;">{segmenter.model_status}</div>
        </div>
        <div>
            <span class="status-pill status-pill-purple">YOLOv11n-seg / Adaptive Edge</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(r"""
    #### 🏗️ Cấu Trúc Khối Xử Lý của Giai Đoạn 2:
    1. **Encoder (Backbone)**: Trích xuất bản đồ đặc trưng cấp cao (Feature Maps) từ ảnh đầu vào.
       - *MobileNetV4 / YOLOv11 C3k2*: Tối ưu hóa sâu trên phần cứng biên với các phép tích chập Depthwise Separable.
    2. **Decoder**: Tái tạo mặt nạ phân đoạn ở kích thước ảnh gốc.
       - *FCN*: Dùng Transposed Convolution để ra pixel mask.
       - *DeepLabV3+*: Dùng Atrous Spatial Pyramid Pooling (ASPP) để nắm bắt ngữ cảnh ở nhiều tỉ lệ khác nhau.
    3. **Morphology & Contour Analysis**:
       - Bóc tách đa giác bao bọc (Polygon Contour).
       - Khớp hình elip tương đương (Ellipse Fitting) để xác định **Trục lớn (Major Axis)** và **Trục bé (Minor Axis)**.
       - Tính toán độ đặc (Solidity = $Area_{contour} / Area_{convex\_hull}$) và tâm quang học (Centroid).
    """)

    sample_imgs = list(SAMPLES_DIR.glob("*.jpg"))
    if sample_imgs:
        st.markdown("---")
        st.markdown("##### 🧪 Thử nghiệm Trích xuất Mặt nạ Phân đoạn trên Ảnh Mẫu:")
        sel_sample = st.selectbox("Chọn ảnh mẫu kiểm tra:", [f.name for f in sample_imgs], key="tab2_sample")
        img_raw = cv2.imread(str(SAMPLES_DIR / sel_sample))
        
        objs = segmenter.segment_image(img_raw)
        
        c_i1, c_i2 = st.columns(2)
        with c_i1:
            st.image(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB), caption="Ảnh gốc đầu vào", use_container_width=True)
        with c_i2:
            mask_vis = img_raw.copy()
            for obj in objs:
                m = obj.get("mask")
                if m is not None:
                    mask_vis[m > 0] = (220, 50, 180)
            st.image(cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB), caption=f"Mặt nạ phân đoạn (Tìm thấy {len(objs)} đối tượng)", use_container_width=True)
            
        if objs:
            top_obj = objs[0]
            st.markdown(f"""
            <div class="telemetry-row">
                <div class="telemetry-box">
                    <div class="telemetry-box-label">Diện tích (Area)</div>
                    <div class="telemetry-box-val">{top_obj['area_px']:.0f} px</div>
                </div>
                <div class="telemetry-box">
                    <div class="telemetry-box-label">Trục lớn (Major)</div>
                    <div class="telemetry-box-val">{top_obj['major_axis_px']:.1f} px</div>
                </div>
                <div class="telemetry-box">
                    <div class="telemetry-box-label">Trục bé (Minor)</div>
                    <div class="telemetry-box-val">{top_obj['minor_axis_px']:.1f} px</div>
                </div>
                <div class="telemetry-box">
                    <div class="telemetry-box-label">Tỉ lệ Khung hình</div>
                    <div class="telemetry-box-val">{top_obj['aspect_ratio']:.2f}</div>
                </div>
                <div class="telemetry-box">
                    <div class="telemetry-box-label">Độ đặc (Solidity)</div>
                    <div class="telemetry-box-val">{top_obj['solidity']:.3f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ==============================================================================
# TAB 4: GIAI ĐOẠN 3 - ĐO LƯỜNG KÍCH THƯỚC & DỰ ĐOÁN KHỐI LƯỢNG
# ==============================================================================
elif menu.startswith("📐 4."):
    st.markdown("""
    <div class="top-hero-banner">
        <h1 class="hero-title">📐 Giai đoạn 3: Đo lường Kích thước & Khối lượng từ Ảnh 2D</h1>
        <div class="hero-subtitle">
            So sánh hai phương pháp: Hình học Chiều sâu Phối cảnh thuần túy vs. Hình học Phối cảnh KÈM kết hợp AI (MLP Regressor).
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    | Đặc điểm | Phương pháp 1: Hình học Chiều sâu Phối cảnh | Phương pháp 2: Hình học Phối cảnh KÈM kết hợp AI |
    |---|---|---|
    | **Nguyên lý** | Ánh xạ pixel -> mm qua tỷ lệ PPM, xấp xỉ thể tích ellipsoid: $V = \\frac{\\pi}{6} L W H$ | Trích xuất 12 chiều đặc trưng hình học + đưa qua Mạng Nơ-ron MLP / Gradient Boosting |
    | **Ưu điểm** | Đơn giản, không cần dữ liệu huấn luyện | Tự động học bù trừ sai số do biến dạng góc chụp, đổ bóng, bề mặt lồi lõm |
    | **Hạn chế** | Sai số 15% - 30% đối với trái cây không tròn đều hoặc bị nghiêng | Cần tối thiểu 5-20 mẫu có nhãn để huấn luyện |
    """)

    st.markdown("---")
    st.markdown("### 🧠 Huấn Luyện Mạng Học Sâu MLP Bù Trừ Sai Số")
    
    X, y, ids = dm.get_training_arrays()
    c_t1, c_t2 = st.columns([2, 1])
    
    with c_t1:
        st.write(f"Dữ liệu huấn luyện hiện tại: **{len(X)}** mẫu đã gán nhãn Ground Truth.")
        if len(X) < 5:
            st.warning("⚠️ Cần tối thiểu 5 mẫu có nhãn để bắt đầu huấn luyện. Hãy chuyển sang Tab 1 hoặc 2 để thêm mẫu!")
        else:
            st.success(f"✓ Tập dữ liệu sẵn sàng ({len(X)} mẫu). Bạn có thể bấm Huấn luyện mô hình ngay.")
            
    with c_t2:
        model_choice = st.selectbox("Kiến trúc Mô hình Hồi quy:", ["mlp", "gradient_boosting"], format_func=lambda x: "Mạng Nơ-ron MLP (32, 16)" if x == "mlp" else "Gradient Boosting Trees")
        if st.button("🚀 Bắt đầu Huấn luyện Mô hình", disabled=(len(X) < 5), use_container_width=True):
            st.session_state.ai_regressor = AIMassRegressor(model_type=model_choice)
            with st.spinner("Đang huấn luyện và tối ưu hàm mất mát..."):
                t_res = st.session_state.ai_regressor.train(X, y)
                if t_res["success"]:
                    st.success(t_res["message"])
                    # Export ONNX immediately
                    ok_o, msg_o = st.session_state.ai_regressor.export_to_onnx()
                    st.info(msg_o)
                    # Re-initialize Edge engine to load new ONNX model
                    st.session_state.edge_engine = EdgeInferenceEngine()
                else:
                    st.error(t_res["message"])

    if regressor.is_trained and regressor.metrics:
        st.markdown("##### 📈 Đánh giá Độ chính xác Sau Huấn luyện (Validation Metrics):")
        m = regressor.metrics
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        c_m1.metric("MAE (Sai số trung bình)", f"{m['mae_grams']:.2f} g")
        c_m2.metric("RMSE (Căn bậc hai sai số)", f"{m['rmse_grams']:.2f} g")
        c_m3.metric("R² Score (Độ tương quan)", f"{m['r2_score']:.3f}")
        c_m4.metric("Sai số phần trăm", f"{m['mean_percentage_error']:.1f} %")

# ==============================================================================
# TAB 5: GIAI ĐOẠN 4 - LƯỢNG TỬ HÓA MÔ HÌNH (QUANTIZATION FP32 -> INT8)
# ==============================================================================
elif menu.startswith("⚡ 5."):
    st.markdown("""
    <div class="top-hero-banner">
        <h1 class="hero-title">⚡ Giai đoạn 4: Lượng tử hóa Mô hình (Model Quantization FP32 → INT8)</h1>
        <div class="hero-subtitle">
            Quá trình ánh xạ trọng số từ số thực 32-bit (FP32) sang số nguyên 8-bit (INT8) bằng phương pháp Post-Training Quantization (PTQ).
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    #### 💡 Tại sao phải lượng tử hóa trên thiết bị biên (Edge AI)?
    - **Giảm dung lượng RAM**: Giúp mô hình nhẹ hơn nhiều lần, chạy mượt mà trên các vi điều khiển hoặc board nhúng (Raspberry Pi, Jetson Nano).
    - **Tăng tốc băng thông bộ nhớ**: Tận dụng các tập lệnh phần cứng INT8 (VNNI của Intel, Tensor Cores của NVIDIA).
    - **Tiết kiệm điện năng**: Giảm nhiệt lượng và tiêu hao năng lượng tối đa khi chạy 24/7.
    """)

    c_q1, c_q2 = st.columns(2)
    with c_q1:
        st.markdown("##### 🔄 Thực thi Lượng tử hóa Post-Training Quantization (PTQ):")
        if st.button("⚡ Thực hiện Lượng tử hóa FP32 ➔ INT8", use_container_width=True):
            with st.spinner("Đang chạy ONNX Runtime Dynamic Quantization..."):
                q_res = ModelQuantizer.quantize_fp32_to_int8()
                if q_res["success"]:
                    st.success(q_res["message"])
                    # Reload edge engine with INT8
                    st.session_state.edge_engine = EdgeInferenceEngine()
                else:
                    st.error(q_res["message"])

    fp32_p = MODEL_CONFIGS["mlp_onnx_fp32"]
    int8_p = MODEL_CONFIGS["mlp_onnx_int8"]
    
    if fp32_p.exists() and int8_p.exists():
        size_fp32 = os.path.getsize(fp32_p) / 1024.0
        size_int8 = os.path.getsize(int8_p) / 1024.0
        reduction = ((size_fp32 - size_int8) / size_fp32) * 100
        
        st.markdown("---")
        st.markdown("##### 📊 So sánh Dung Lượng & Tốc Độ Thực Thi:")
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown(f"""
            <div class="metrology-card metrology-card-emerald">
                <div class="card-label">Dung lượng Tệp Mô hình</div>
                <div style="font-size:1.15rem; margin-top:0.5rem; font-family:'JetBrains Mono';">
                    FP32: <b>{size_fp32:.1f} KB</b> ➔ INT8: <b style="color:#4ade80;">{size_int8:.1f} KB</b>
                </div>
                <div class="card-footer-sub">Tỉ lệ nén tiết kiệm: <b style="color:#38bdf8;">{reduction:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_res2:
            if st.button("⏱️ Chạy Benchmark Đo Độ Trễ (Latency)", use_container_width=True):
                bench = ModelQuantizer.compare_inference_performance(iterations=300)
                if bench["success"]:
                    st.markdown(f"""
                    <div class="metrology-card">
                        <div class="card-label">Độ trễ suy luận trung bình (300 iterations)</div>
                        <div style="font-size:1rem; margin-top:0.5rem; font-family:'JetBrains Mono';">
                            FP32: <b>{bench['latency_fp32_ms']:.4f} ms</b> ({bench['fps_fp32']:.0f} FPS)<br>
                            INT8: <b style="color:#4ade80;">{bench['latency_int8_ms']:.4f} ms</b> ({bench['fps_int8']:.0f} FPS)
                        </div>
                        <div class="card-footer-sub">Tăng tốc: <b>{bench['speedup_factor']:.2f}x</b></div>
                    </div>
                    """, unsafe_allow_html=True)

# ==============================================================================
# TAB 6: GIAI ĐOẠN 5 - TĂNG TỐC PHẦN CỨNG & EDGE AI
# ==============================================================================
elif menu.startswith("🚀 6."):
    st.markdown("""
    <div class="top-hero-banner">
        <h1 class="hero-title">🚀 Giai đoạn 5: Chuyển đổi & Tăng tốc Suy luận trên Phần cứng</h1>
        <div class="hero-subtitle">
            Triển khai mã nguồn mở ONNX, kích hoạt OpenVINO Execution Provider (Intel) hoặc TensorRT (NVIDIA) để đạt độ trễ mili-giây.
        </div>
    </div>
    """, unsafe_allow_html=True)

    edge = st.session_state.edge_engine
    
    c_e1, c_e2 = st.columns(2)
    with c_e1:
        st.markdown(f"""
        <div class="metrology-card metrology-card-emerald">
            <div class="card-label">Execution Provider Đang Kích Hoạt</div>
            <div class="card-value-display" style="color:#38bdf8; font-size:1.6rem;">{edge.active_provider}</div>
            <div class="card-footer-sub">Tối ưu hóa bởi ONNX Runtime Engine</div>
        </div>
        """, unsafe_allow_html=True)
    with c_e2:
        st.markdown(f"""
        <div class="metrology-card">
            <div class="card-label">Các Execution Providers Có sẵn trên máy</div>
            <div style="margin-top:0.5rem; font-weight:600; color:#e2e8f0; font-family:'JetBrains Mono';">
                {', '.join(edge.available_providers)}
            </div>
            <div class="card-footer-sub">OpenVINO / TensorRT / CUDA / CPU</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    #### ⚙️ Khung Tăng Tốc Phần Cứng Chuẩn Công Nghiệp:
    - **OpenVINO (Intel)**: Tự động phân bổ luồng xử lý trên CPU Intel Core, iGPU Iris Xe hoặc chip NPU chuyên dụng.
    - **TensorRT (NVIDIA)**: Độc quyền của NVIDIA, thực hiện kết hợp các tầng (Layer Fusion) và lượng tử hóa FP16/INT8 trực tiếp trên Tensor Cores.
    - **CPU Execution Provider**: Chạy đa luồng SIMD (AVX2/AVX-512) mượt mà trên mọi thiết bị mà không cần card rời.
    """)

    st.markdown("---")
    st.markdown("##### ⏱️ Chạy Benchmark Tốc Độ Thực Tế:")
    if st.button("🔥 Khởi chạy Benchmark Hiệu Năng Phần Cứng", use_container_width=True):
        with st.spinner("Đang chạy kiểm tra độ trễ đa vòng lặp..."):
            b_res = edge.benchmark_providers(iterations=500)
            if b_res["success"]:
                st.write("Kết quả kiểm tra chi tiết trên hệ thống hiện tại:")
                for prov, data in b_res["benchmarks"].items():
                    if data["latency_ms"] is not None:
                        st.success(f"**{prov}**: Độ trễ {data['latency_ms']:.4f} ms | Tốc độ: {data['fps']:.1f} FPS (Trạng thái: {data['status']})")
                    else:
                        st.info(f"**{prov}**: {data['status']}")
            else:
                st.error(b_res["message"])
