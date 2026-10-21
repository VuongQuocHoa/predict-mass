"""
Real-time Camera Detection & Mass Estimation Engine (Industrial Pipelined).
Industrial Vision Architecture:
- Non-blocking Hardware Frame Grabber: Sensor-speed acquisition (30-60+ FPS)
- Decoupled Metrology & AI Inference Worker: Asynchronous background worker
- Shortest-Process Delay Principle: Display stream latency is decoupled from AI
  latency and depends strictly on the fastest stage (Camera sensor ~16-33ms).
- Parallel Scale/ArUco Tracking & Object Segmentation: T_ai = max(T_aruco, T_seg).

Keyboard Shortcuts:
- 'q' or ESC: Quit
- 's': Save current frame & store sample into dataset registry
- 'c': Toggle object class (apple, orange, banana, lemon, mango, etc.)
- 'z': Recalibrate Camera Distance Z
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import time
import numpy as np
from pathlib import Path
from config import OBJECT_DENSITIES, SAMPLES_DIR
from core.industrial_pipeline import IndustrialAsyncPipeline
from core.data_manager import DatasetManager

def draw_hud(frame, results, aruco_info, object_class, stream_fps, ai_fps, ai_latency_ms, calib_status):
    """Draws an industrial-grade telemetry HUD overlay on the live camera frame."""
    h, w = frame.shape[:2]
    fastest_delay_ms = 1000.0 / max(1.0, stream_fps)
    
    # 1. Overlay pass for top and bottom control bars
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 58), (15, 23, 42), -1)
    cv2.rectangle(overlay, (0, h - 35), (w, h), (15, 23, 42), -1)

    # Fill segmentation masks / contours into overlay
    for item in results:
        mask = item["morphology"].get("mask")
        if mask is not None:
            overlay[mask > 0] = (210, 80, 50) # Cyber coral accent
        elif "contour" in item["morphology"]:
            cv2.drawContours(overlay, [item["morphology"]["contour"]], -1, (210, 80, 50), -1)

    # Blend overlay once with 0.70 alpha
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    
    # Top Title & Pipeline Status
    cv2.putText(frame, "INDUSTRIAL DECOUPLED METROLOGY & AI MASS ESTIMATION", (15, 22), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    
    # Primary Telemetry: Stream FPS (Shortest process path) + AI Worker FPS & Latency
    fps_text = f"STREAM: {stream_fps:.1f} FPS ({fastest_delay_ms:.1f}ms) | AI: {ai_fps:.1f} FPS ({ai_latency_ms:.1f}ms)"
    cv2.putText(frame, fps_text, (w - 480, 22), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (74, 222, 128), 2)
                
    # Sub-status: Object class & Calibration & Architecture badge
    sub_text = f"CLASS: {object_class.upper()} | {calib_status} | ARCH: ZERO-BOTTLENECK PIPELINE"
    cv2.putText(frame, sub_text, (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (203, 213, 225), 1)
    
    # Bottom instructions bar text
    instruct = "[S] Luu mau  |  [C] Doi loai qua  |  [Z] Nhap Z-depth  |  [Q] Thoat"
    cv2.putText(frame, instruct, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (148, 163, 184), 1)

    # Draw ArUco marker box if present
    if aruco_info.get("found") and aruco_info.get("corners") is not None:
        corners = aruco_info["corners"].astype(int)
        cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
        cv2.putText(frame, f"ArUco Scale ({aruco_info['ppm']:.2f} px/mm)", 
                    (corners[0][0], max(20, corners[0][1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw Detected Objects Contours and HUD Telemetry Tags
    for item in results:
        bbox = item["morphology"]["bbox_xywh"]
        cnt = item["morphology"]["contour"]
        x, y, bw, bh = bbox
        
        # Draw contour & bounding box
        cv2.drawContours(frame, [cnt], -1, (244, 114, 182), 2)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (56, 189, 248), 1)
        
        # Floating telemetry card next to object
        ai_mass = item["ai_predicted_mass_g"]
        geom = item["geometry"]
        
        card_x = min(w - 220, max(10, x))
        card_y = max(65, y - 60)
        
        # Translucent tag card
        cv2.rectangle(frame, (card_x, card_y), (card_x + 210, card_y + 55), (15, 23, 42), -1)
        cv2.rectangle(frame, (card_x, card_y), (card_x + 210, card_y + 55), (56, 189, 248), 1)
        
        mass_txt = f"MASS: {ai_mass:.1f} g"
        cv2.putText(frame, mass_txt, (card_x + 8, card_y + 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (74, 222, 128), 2)
                    
        dim_txt = f"{geom['length_mm']:.0f}x{geom['width_mm']:.0f} mm | V:{geom['volume_cm3']:.0f}cm3"
        cv2.putText(frame, dim_txt, (card_x + 8, card_y + 44), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1)

    return frame

def run_realtime_camera(camera_index: int = 0, initial_class: str = "apple"):
    print(f"[*] Khởi động Industrial Decoupled Vision Pipeline (Camera Index {camera_index})...")
    
    pipeline = IndustrialAsyncPipeline(
        camera_index=camera_index,
        initial_class=initial_class,
        width=1280,
        height=720
    )
    
    if not pipeline.start():
        print(f"[!] Không thể mở Camera {camera_index}. Vui lòng kiểm tra kết nối thiết bị.")
        return

    data_manager = DatasetManager()
    class_keys = list(OBJECT_DENSITIES.keys())
    current_class_idx = class_keys.index(initial_class) if initial_class in class_keys else 0
    
    window_name = "Industrial AI Metrology (Decoupled Pipeline - Press 'Q' to Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 600)
    
    print("[*] Industrial Pipeline đã kích hoạt!")
    print("    - Cơ chế: Thu nhận phần cứng (Grabber) độc lập với Xử lý AI")
    print("    - Độ trễ hiển thị: Bị chi phối bởi tiến trình ngắn nhất (~16-33ms)")
    print("    - Phím 'S': Chụp và lưu mẫu kèm khối lượng")
    print("    - Phím 'C': Đổi loại quả")
    print("    - Phím 'Q': Thoát chương trình")

    try:
        while True:
            # Fetch latest fresh camera frame and concurrent AI state
            frame, results, aruco_info, calib_status, stream_fps, ai_fps, ai_latency_ms = pipeline.get_render_state()
            
            if frame is None:
                time.sleep(0.005)
                continue
                
            curr_class = class_keys[current_class_idx]
            
            # Render HUD overlay at full display/sensor speed
            display_frame = draw_hud(
                frame=frame,
                results=results,
                aruco_info=aruco_info,
                object_class=curr_class,
                stream_fps=stream_fps,
                ai_fps=ai_fps,
                ai_latency_ms=ai_latency_ms,
                calib_status=calib_status
            )
            cv2.imshow(window_name, display_frame)

            # Keyboard interactions (non-blocking)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # Q or ESC
                break
            elif key == ord('c'):
                current_class_idx = (current_class_idx + 1) % len(class_keys)
                new_class = class_keys[current_class_idx]
                pipeline.set_object_class(new_class)
                print(f"[i] Đã chuyển loại quả sang: {new_class.upper()}")
            elif key == ord('s'):
                # Save snapshot
                save_ts = int(time.time())
                save_path = SAMPLES_DIR / f"industrial_{curr_class}_{save_ts}.jpg"
                cv2.imwrite(str(save_path), frame)
                print(f"\n[+] Đã lưu ảnh chụp: {save_path.name}")
                
                if results:
                    best_obj = results[0]
                    geom = best_obj["geometry"]
                    est_ai = best_obj["ai_predicted_mass_g"]
                    print(f"    - Kích thước: {geom['length_mm']:.1f} x {geom['width_mm']:.1f} x {geom['height_mm']:.1f} mm")
                    print(f"    - Thể tích: {geom['volume_cm3']:.1f} cm3")
                    print(f"    - Khối lượng ước tính: {est_ai:.1f}g")
                    
                    data_manager.add_sample(
                        image_path=str(save_path),
                        object_class=curr_class,
                        feature_vector=best_obj["feature_vector"],
                        geom_info=geom,
                        actual_weight_g=est_ai,
                        metadata={"source": "industrial_decoupled_stream"}
                    )
                    print(f"    [OK] Đã lưu mẫu vào Registry data/dataset.json!")
            elif key == ord('z'):
                pipeline.geom_engine.calibrate_with_distance(500.0, 800.0)
                print("\n[?] Hiệu chuẩn Z: Sử dụng mặc định Z=500mm.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[*] Đã tắt Industrial Pipeline an toàn.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Industrial Decoupled Real-time Camera Mass Estimation")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--class_name", type=str, default="apple", help="Initial fruit class")
    args = parser.parse_args()
    
    run_realtime_camera(camera_index=args.cam, initial_class=args.class_name)
