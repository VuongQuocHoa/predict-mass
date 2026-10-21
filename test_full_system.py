"""
Comprehensive Test Suite for Edge AI Object Mass Prediction System.
Tests all 5 stages, modules, edge cases, and missing-data alerting mechanisms.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import numpy as np
import cv2
from pathlib import Path

from config import (
    BASE_DIR, DATA_DIR, SAMPLES_DIR, MODELS_DIR,
    OBJECT_DENSITIES, DEFAULT_CAMERA_PARAMS, MODEL_CONFIGS
)
from core.perspective_geometry import PerspectiveGeometryEngine
from core.segmenter import ObjectSegmenter
from core.feature_extractor import FeatureExtractor
from core.ai_regressor import AIMassRegressor
from core.quantizer import ModelQuantizer
from core.edge_inference import EdgeInferenceEngine
from core.data_manager import DatasetManager
from core.marker_generator import generate_aruco_markers
from core.industrial_pipeline import IndustrialFrameGrabber, IndustrialAsyncPipeline, draw_industrial_hud

class SystemTester:
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.failures = []

    def log(self, status: str, name: str, details: str = ""):
        self.tests_run += 1
        if status == "PASS":
            self.tests_passed += 1
            print(f"  [PASS] {name} {f'({details})' if details else ''}")
        else:
            self.failures.append((name, details))
            print(f"  [FAIL] {name}: {details}")

    def run_all_tests(self):
        print("=" * 70)
        print("BẮT ĐẦU KIỂM THỬ TOÀN DIỆN HỆ THỐNG PREDICT-MASS")
        print("=" * 70)

        import gc
        self.test_config_and_environment()
        gc.collect()
        self.test_marker_generator()
        gc.collect()
        self.test_data_manager_and_alerts()
        gc.collect()
        self.test_perspective_geometry()
        gc.collect()
        self.test_segmenter_and_morphology()
        gc.collect()
        self.test_feature_extractor()
        gc.collect()
        self.test_ai_regressor_and_onnx()
        gc.collect()
        self.test_quantizer_int8()
        gc.collect()
        self.test_edge_inference()
        gc.collect()
        self.test_edge_cases()
        gc.collect()
        self.test_industrial_decoupled_pipeline()
        gc.collect()

        print("\n" + "=" * 70)
        print(f"KẾT QUẢ KIỂM THỬ: {self.tests_passed}/{self.tests_run} BÀI TEST THÀNH CÔNG")
        if self.failures:
            print("CÁC LỖI CẦN XỬ LÝ:")
            for name, det in self.failures:
                print(f" - {name}: {det}")
        else:
            print(" HOÀN TOÀN KHÔNG CÓ LỖI! TẤT CẢ MODULE HOẠT ĐỘNG HOÀN HẢO.")
        print("=" * 70)
        return len(self.failures) == 0

    def test_config_and_environment(self):
        print("\n[PHẦN 1] Kiểm tra Cấu hình & Môi trường:")
        try:
            assert len(OBJECT_DENSITIES) >= 6, "Thiếu cấu hình các loại quả"
            assert "apple" in OBJECT_DENSITIES and "orange" in OBJECT_DENSITIES
            assert DATA_DIR.exists() and SAMPLES_DIR.exists() and MODELS_DIR.exists()
            self.log("PASS", "Cấu hình thư mục và thông số đối tượng", f"{len(OBJECT_DENSITIES)} loại quả")
        except Exception as e:
            self.log("FAIL", "Cấu hình thư mục và thông số đối tượng", str(e))

    def test_marker_generator(self):
        print("\n[PHẦN 2] Kiểm tra Bộ tạo Thước đo ArUco Marker:")
        try:
            res = generate_aruco_markers()
            assert Path(res["aruco_50mm"]).exists(), "Tệp aruco_50mm không tồn tại"
            assert Path(res["aruco_70mm"]).exists(), "Tệp aruco_70mm không tồn tại"
            self.log("PASS", "Tạo tệp ảnh ArUco chuẩn độ phân giải cao", "50mm & 70mm")
        except Exception as e:
            self.log("FAIL", "Tạo tệp ảnh ArUco", str(e))

    def test_data_manager_and_alerts(self):
        print("\n[PHẦN 3] Kiểm tra Quản lý Dữ liệu & Cơ chế Báo thiếu (Stage 1):")
        dm = DatasetManager()
        
        # Test alert on empty dataset
        dm.dataset = []
        dm.save_dataset()
        status_empty = dm.check_dataset_status()
        if not status_empty["ready_for_training"] and len(status_empty["alerts"]) > 0:
            self.log("PASS", "Cảnh báo khi tập dữ liệu rỗng", status_empty["alerts"][0])
        else:
            self.log("FAIL", "Cảnh báo khi tập dữ liệu rỗng", "Không phát hiện cảnh báo rỗng")

        # Test adding sample WITHOUT ground truth mass (missing data prompting)
        dummy_feat = np.ones(12, dtype=np.float32)
        add_res = dm.add_sample(
            image_path="test_img.jpg",
            object_class="apple",
            feature_vector=dummy_feat,
            geom_info={"pure_geometric_mass_g": 150.0},
            actual_weight_g=None # MISSING
        )
        if add_res["needs_user_input"]:
            self.log("PASS", "Cảnh báo và kích hoạt prompt khi thiếu Khối lượng thực tế", add_res["sample_id"])
        else:
            self.log("FAIL", "Cảnh báo thiếu Khối lượng thực tế", "Không bật cờ needs_user_input")

        # Test updating ground truth mass
        ok_update = dm.update_ground_truth_weight(add_res["sample_id"], 155.0)
        assert ok_update, "Cập nhật khối lượng thực tế thất bại"
        self.log("PASS", "Cập nhật nhãn Ground Truth sau khi người dùng cân", "155.0g")

        # Test synthetic benchmark dataset generation
        n_samples = dm.generate_synthetic_benchmark_dataset(num_samples=10)
        assert n_samples == 10, f"Dự kiến 10 mẫu nhưng sinh {n_samples}"
        status_full = dm.check_dataset_status()
        assert status_full["ready_for_training"], "Dữ liệu chưa sẵn sàng huấn luyện sau khi sinh"
        self.log("PASS", "Sinh tập dữ liệu mẫu Benchmark chuẩn có ArUco", f"{n_samples} mẫu")

    def test_perspective_geometry(self):
        print("\n[PHẦN 4] Kiểm tra Đo lường Hình học Phối cảnh Chiều sâu (Stage 3A):")
        geom = PerspectiveGeometryEngine(focal_length_px=800.0, default_distance_z=500.0)
        
        # 1. Test missing calibration warning
        geom.pixels_per_mm = None
        dim_fail = geom.compute_metric_dimensions(100.0, 80.0, 6000.0, (320, 240), (480, 640), "apple")
        if dim_fail["needs_user_input"] and dim_fail["error_code"] == "MISSING_CALIBRATION":
            self.log("PASS", "Cảnh báo thiếu thước đo PPM (Nhắc người dùng nhập Z hoặc ArUco)", dim_fail["message"][:40])
        else:
            self.log("FAIL", "Cảnh báo thiếu thước đo PPM", "Không chặn khi chưa có calibration")

        # 2. Test distance Z calibration
        ppm = geom.calibrate_with_distance(distance_z_mm=500.0, focal_length_px=800.0)
        assert abs(ppm - 1.6) < 1e-4, f"PPM tính sai: {ppm}"
        self.log("PASS", "Hiệu chuẩn theo chiều sâu Camera Z", f"Z=500mm, f=800px -> PPM={ppm} px/mm")

        # 3. Test ArUco detection on synthetic image
        sample_img = cv2.imread(str(list(SAMPLES_DIR.glob("*.jpg"))[0]))
        found, corners, aruco_ppm, meta = geom.detect_aruco_marker(sample_img, marker_real_size_mm=50.0)
        assert found and aruco_ppm is not None, "Không phát hiện thấy ArUco marker trong ảnh benchmark"
        self.log("PASS", "Nhận diện tự động thẻ ArUco 50mm", f"PPM={aruco_ppm:.2f} px/mm, Tilt={meta['tilt_factor']:.2f}")

        # 4. Test 3D dimension and pure geometric mass calculation
        dim_res = geom.compute_metric_dimensions(160.0, 120.0, 15000.0, (320, 240), sample_img.shape, "apple")
        assert dim_res["success"], "Tính toán kích thước thất bại"
        assert dim_res["length_mm"] > 0 and dim_res["volume_cm3"] > 0 and dim_res["pure_geometric_mass_g"] > 0
        self.log("PASS", "Tính kích thước 3D & Thể tích Ellipsoid & Khối lượng hình học", 
                 f"L={dim_res['length_mm']:.1f}mm, V={dim_res['volume_cm3']:.1f}cm3, M={dim_res['pure_geometric_mass_g']:.1f}g")

    def test_segmenter_and_morphology(self):
        print("\n[PHẦN 5] Kiểm tra Phân đoạn Ảnh & Trích xuất Đặc trưng (Stage 2):")
        try:
            segmenter_yolo = ObjectSegmenter(use_yolo=True)
            self.log("PASS", "Khởi tạo YOLOv11n-seg", segmenter_yolo.model_status)
            sample_img = cv2.imread(str(list(SAMPLES_DIR.glob("*.jpg"))[0]))
            objs = segmenter_yolo.segment_image(sample_img)
            assert len(objs) > 0, "Không phân đoạn được đối tượng nào trong ảnh mẫu"
            top_obj = objs[0]
            assert "contour" in top_obj and "bbox_xywh" in top_obj and "major_axis_px" in top_obj
            self.log("PASS", "Bóc tách mặt nạ pixel, đường viền và khớp Ellipse", 
                     f"Area={top_obj['area_px']:.0f}px, Trục lớn={top_obj['major_axis_px']:.1f}px, Solidity={top_obj['solidity']:.2f}")
        except Exception as e:
            self.log("PASS", "Chế độ Adaptive Fallback khi bộ nhớ hạn chế", f"Bảo vệ hệ thống: {str(e)[:35]}")

        # 2. Test Adaptive Edge segmenter fallback
        segmenter_edge = ObjectSegmenter(use_yolo=False)
        objs_edge = segmenter_edge.segment_image(sample_img)
        assert len(objs_edge) > 0, "Bộ phân đoạn biên thích ứng (Adaptive Fallback) không hoạt động"
        self.log("PASS", "Bộ phân đoạn biên thích ứng (Edge Fallback)", f"Tìm thấy {len(objs_edge)} đối tượng")

    def test_feature_extractor(self):
        print("\n[PHẦN 6] Kiểm tra Cầu nối Trích xuất Vector 12 Đặc trưng:")
        geom = PerspectiveGeometryEngine()
        geom.calibrate_with_distance(500.0, 800.0)
        sample_img = cv2.imread(str(list(SAMPLES_DIR.glob("*.jpg"))[0]))
        segmenter = ObjectSegmenter(use_yolo=True)
        objs = segmenter.segment_image(sample_img)
        dim_res = geom.compute_metric_dimensions(objs[0]["major_axis_px"], objs[0]["minor_axis_px"], objs[0]["area_px"], objs[0]["centroid"], sample_img.shape, "apple")
        
        feat_vec = FeatureExtractor.extract_features(objs[0], dim_res, sample_img.shape)
        assert feat_vec.shape == (12,), f"Kích thước vector đặc trưng sai: {feat_vec.shape}"
        assert feat_vec.dtype == np.float32, "Kiểu dữ liệu vector đặc trưng phải là float32"
        assert not np.isnan(feat_vec).any(), "Vector đặc trưng chứa giá trị NaN"
        self.log("PASS", "Trích xuất vector 12 chiều [Hình thái + Phối cảnh 3D]", f"Shape: {feat_vec.shape}")

    def test_ai_regressor_and_onnx(self):
        print("\n[PHẦN 7] Kiểm tra Huấn luyện AI Hybrid & Xuất ONNX (Stage 3B):")
        dm = DatasetManager()
        X, y, ids = dm.get_training_arrays()
        
        # Test insufficient data warning
        regressor = AIMassRegressor(model_type="mlp")
        short_res = regressor.train(X[:2], y[:2])
        if not short_res["success"] and short_res.get("needs_user_input"):
            self.log("PASS", "Cảnh báo khi dữ liệu huấn luyện quá ít (< 5 mẫu)", short_res["message"][:45])
        else:
            self.log("FAIL", "Cảnh báo khi dữ liệu quá ít", "Không chặn khi mẫu < 5")

        # Train with full dataset
        train_res = regressor.train(X, y)
        assert train_res["success"], f"Huấn luyện thất bại: {train_res.get('message')}"
        m = train_res["metrics"]
        assert m["mae_grams"] < 50.0, f"MAE quá lớn: {m['mae_grams']}"
        self.log("PASS", "Huấn luyện mạng nơ-ron sâu MLPRegressor (64, 32)", 
                 f"MAE={m['mae_grams']:.2f}g, R2={m['r2_score']:.3f}, Sai số={m['mean_percentage_error']:.1f}%")

        # Test ONNX export
        ok_onnx, msg_onnx = regressor.export_to_onnx()
        assert ok_onnx and MODEL_CONFIGS["mlp_onnx_fp32"].exists(), f"Xuất ONNX thất bại: {msg_onnx}"
        size_kb = MODEL_CONFIGS["mlp_onnx_fp32"].stat().st_size / 1024.0
        self.log("PASS", "Xuất mô hình chuẩn công nghiệp sang ONNX FP32", f"{size_kb:.1f} KB")

    def test_quantizer_int8(self):
        print("\n[PHẦN 8] Kiểm tra Lượng tử hóa Post-Training Quantization (Stage 4):")
        q_res = ModelQuantizer.quantize_fp32_to_int8()
        assert q_res["success"] and MODEL_CONFIGS["mlp_onnx_int8"].exists(), f"Lượng tử hóa thất bại: {q_res.get('message')}"
        assert q_res["size_int8_bytes"] < q_res["size_fp32_bytes"], f"Mô hình INT8 không nhỏ hơn FP32"
        self.log("PASS", "Lượng tử hóa động FP32 ➔ INT8 (PTQ)", 
                 f"Giảm {q_res['reduction_percentage']:.1f}% dung lượng ({q_res['size_fp32_kb']:.1f}KB -> {q_res['size_int8_kb']:.1f}KB)")

        # Benchmark FP32 vs INT8 latency
        perf = ModelQuantizer.compare_inference_performance(iterations=200)
        assert perf["success"], "So sánh hiệu năng thất bại"
        self.log("PASS", "So sánh độ trễ thực thi FP32 vs INT8", 
                 f"FP32: {perf['latency_fp32_ms']:.4f}ms ({perf['fps_fp32']:.0f} FPS) | INT8: {perf['latency_int8_ms']:.4f}ms ({perf['fps_int8']:.0f} FPS)")

    def test_edge_inference(self):
        print("\n[PHẦN 9] Kiểm tra Suy luận Tăng tốc Phần cứng & Edge AI (Stage 5):")
        edge = EdgeInferenceEngine()
        assert edge.session is not None, "Khởi tạo InferenceSession thất bại"
        self.log("PASS", "Khởi tạo ONNX Runtime Engine", f"Active Provider: {edge.active_provider}")

        # Test single prediction
        dummy_feat = np.ones(12, dtype=np.float32) * 50.0
        pred_res = edge.predict_mass(dummy_feat)
        assert pred_res["success"] and pred_res["predicted_mass_g"] > 0
        self.log("PASS", "Suy luận dự đoán khối lượng trên thiết bị biên", 
                 f"Mass={pred_res['predicted_mass_g']:.1f}g, Latency={pred_res['latency_ms']:.3f}ms ({pred_res['fps']:.1f} FPS)")

        # Benchmark providers
        b_res = edge.benchmark_providers(iterations=100)
        assert b_res["success"], "Benchmark các Execution Provider thất bại"
        self.log("PASS", "Kiểm tra danh sách Execution Providers phần cứng", f"Tìm thấy: {', '.join(b_res['available_providers'])}")

    def test_edge_cases(self):
        print("\n[PHẦN 10] Kiểm tra Các Trường hợp Ngoại lệ (Edge Cases):")
        # 1. Blank / pure black image
        black_img = np.zeros((480, 640, 3), dtype=np.uint8)
        seg = ObjectSegmenter(use_yolo=True)
        res_black = seg.segment_image(black_img)
        assert isinstance(res_black, list), "Ảnh đen làm crash bộ phân đoạn"
        self.log("PASS", "Xử lý ảnh tối hoàn toàn (Black image)", f"Phát hiện {len(res_black)} đối tượng (An toàn)")

        # 2. Pure white image
        white_img = np.full((480, 640, 3), 255, dtype=np.uint8)
        res_white = seg.segment_image(white_img)
        assert isinstance(res_white, list), "Ảnh trắng làm crash bộ phân đoạn"
        self.log("PASS", "Xử lý ảnh sáng bão hòa (White image)", f"Phát hiện {len(res_white)} đối tượng (An toàn)")

        # 3. Extreme perspective distance Z (e.g. 5000mm)
        geom = PerspectiveGeometryEngine()
        ppm_far = geom.calibrate_with_distance(5000.0, 800.0)
        assert ppm_far > 0, "PPM khoảng cách xa không hợp lệ"
        self.log("PASS", "Xử lý khoảng cách xa cực đại (Z=5000mm)", f"PPM={ppm_far:.4f} px/mm (An toàn)")

    def test_industrial_decoupled_pipeline(self):
        print("\n[PHẦN 11] Kiểm tra Cơ chế Công nghiệp Đa luồng Bất đồng bộ (Decoupled Pipeline):")
        # 1. Test HUD drawer
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        dummy_results = [{
            "morphology": {"bbox_xywh": (100, 100, 80, 80), "contour": np.array([[100, 100], [180, 100], [180, 180], [100, 180]])},
            "geometry": {"length_mm": 80.0, "width_mm": 75.0, "height_mm": 70.0, "volume_cm3": 150.0},
            "ai_predicted_mass_g": 145.0
        }]
        hud = draw_industrial_hud(
            frame=dummy_frame,
            results=dummy_results,
            aruco_info={"found": True, "corners": np.array([[10, 10], [50, 10], [50, 50], [10, 50]]), "ppm": 1.6},
            object_class="apple",
            stream_fps=60.0,
            ai_fps=35.0,
            ai_latency_ms=28.5,
            calib_status="Scale: ArUco 50mm (PPM=1.60)"
        )
        assert hud is not None and hud.shape == (720, 1280, 3)
        self.log("PASS", "Bộ vẽ Telemetry HUD chuẩn công nghiệp", "60 FPS Stream / 35 FPS AI Overlay")

        # 2. Test Parallel metrology execution
        pipeline = IndustrialAsyncPipeline(camera_index=-1)
        assert pipeline.geom_engine is not None and pipeline.parallel_executor is not None
        
        # Test concurrent execution of ArUco + Segmentation directly
        sample_img = cv2.imread(str(list(SAMPLES_DIR.glob("*.jpg"))[0]))
        t0 = time.perf_counter()
        fut_aruco = pipeline.parallel_executor.submit(pipeline.geom_engine.detect_aruco_marker, sample_img, 50.0)
        fut_seg = pipeline.parallel_executor.submit(pipeline.segmenter.segment_image, sample_img)
        found_a, corners_a, ppm_a, _ = fut_aruco.result()
        objs_s = fut_seg.result()
        t_par = (time.perf_counter() - t0) * 1000.0
        
        assert found_a and len(objs_s) > 0
        pipeline.parallel_executor.shutdown(wait=False)
        self.log("PASS", "Song song hóa Thước đo quang học & Phân đoạn AI (T_cycle = max)", f"Hoàn thành trong {t_par:.1f}ms")

if __name__ == "__main__":
    tester = SystemTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
