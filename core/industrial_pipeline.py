"""
Industrial Decoupled Asynchronous Vision Pipeline.
Architecture:
- Stage 1: Hardware-Acquisition Thread (Frame Grabber / Producer)
  * Runs at maximum hardware sensor rate (30-60+ FPS).
  * Direct hardware buffer draining with zero stale-frame latency.
- Stage 2: Industrial Metrology & AI Inference Worker (Asynchronous Consumer)
  * Concurrently executes Scale/ArUco Tracking and Object Segmentation.
  * T_vision = max(T_aruco, T_seg) instead of T_aruco + T_seg.
  * Computes 3D geometry & accelerated ONNX INT8 mass prediction.
- Stage 3: Low-Latency Display & Telemetry Dispatcher (Zero Bottleneck)
  * System delay is decoupled: Display path latency is determined strictly
    by the shortest process (Camera frame interval ~16-33ms), NOT the sum
    of capture + AI inference!
"""
import time
import threading
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from config import DEFAULT_CAMERA_PARAMS, OBJECT_DENSITIES
from core.perspective_geometry import PerspectiveGeometryEngine
from core.segmenter import ObjectSegmenter
from core.feature_extractor import FeatureExtractor
from core.edge_inference import EdgeInferenceEngine


class IndustrialFrameGrabber:
    """
    Dedicated high-priority hardware frame acquisition thread.
    Continuously flushes the OpenCV/OS driver internal ring buffer so that
    retrieved frames always represent the freshest visual instant with zero lag.
    """
    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720, target_fps: int = 60):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        
        self.cap = None
        self.is_running = False
        self.thread = None
        
        self._lock = threading.Lock()
        self._latest_frame = None
        self._frame_timestamp = 0.0
        self._frame_count = 0
        
        # FPS telemetry
        self.stream_fps = 0.0
        self._fps_prev_time = time.time()
        self._fps_frame_counter = 0

    def start(self) -> bool:
        # Open camera with DirectShow backend for lowest latency on Windows
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)
            
        if not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True, name="IndustrialFrameGrabber")
        self.thread.start()
        return True

    def _capture_loop(self):
        while self.is_running and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            now = time.time()
            with self._lock:
                self._latest_frame = frame
                self._frame_timestamp = now
                self._frame_count += 1

            # Compute actual camera acquisition FPS
            self._fps_frame_counter += 1
            elapsed = now - self._fps_prev_time
            if elapsed >= 0.5:
                self.stream_fps = self._fps_frame_counter / elapsed
                self._fps_prev_time = now
                self._fps_frame_counter = 0

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Returns (latest_frame, timestamp) in a non-blocking instant read."""
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0
            return self._latest_frame.copy(), self._frame_timestamp

    def stop(self):
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class IndustrialAsyncPipeline:
    """
    Decoupled Industrial Vision Pipeline.
    Runs camera acquisition, concurrent visual metrology (ArUco + Segmentation),
    and HUD rendering on separate asynchronous paths to prevent bottlenecking.
    """
    def __init__(
        self,
        camera_index: int = 0,
        initial_class: str = "apple",
        width: int = 1280,
        height: int = 720
    ):
        self.camera_index = camera_index
        self.current_class = initial_class
        self.width = width
        self.height = height

        # Components
        self.grabber = IndustrialFrameGrabber(camera_index=camera_index, width=width, height=height)
        self.geom_engine = PerspectiveGeometryEngine(
            focal_length_px=DEFAULT_CAMERA_PARAMS["focal_length_px"],
            default_distance_z=DEFAULT_CAMERA_PARAMS["default_distance_z_mm"]
        )
        self.segmenter = ObjectSegmenter(use_yolo=True)
        self.edge_engine = EdgeInferenceEngine()

        # Thread pool for concurrent ArUco tracking and Object Segmentation
        self.parallel_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="MetrologyWorker")

        # Asynchronous worker thread
        self.is_running = False
        self.worker_thread = None

        # Thread-safe results exchange
        self._results_lock = threading.Lock()
        self._latest_results: List[Dict[str, Any]] = []
        self._aruco_info: Dict[str, Any] = {"found": False, "corners": None, "ppm": 1.6}
        self._calib_status: str = "Scale: Camera Height Z=500mm (Fixed Plane)"
        
        # Telemetry & Performance Metrics
        self.ai_fps: float = 0.0
        self.ai_latency_ms: float = 0.0
        self.fastest_stage_delay_ms: float = 16.6
        self.total_processed_frames: int = 0

    def start(self) -> bool:
        if not self.grabber.start():
            return False

        # Calibrate default plane
        self.geom_engine.calibrate_with_distance(500.0, 800.0)

        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="IndustrialAIWorker")
        self.worker_thread.start()
        return True

    def _worker_loop(self):
        """
        Asynchronous Metrology & AI inference worker.
        Fetches the latest available frame, concurrently processes ArUco and Segmentation,
        and posts telemetry updates without slowing down camera capture or rendering.
        """
        last_processed_ts = 0.0
        ai_frame_count = 0
        fps_start_time = time.time()

        while self.is_running:
            frame, frame_ts = self.grabber.get_latest_frame()
            if frame is None or frame_ts <= last_processed_ts:
                time.sleep(0.002)
                continue

            last_processed_ts = frame_ts
            t0 = time.perf_counter()

            # -------------------------------------------------------------
            # INDUSTRIAL CONCURRENCY:
            # Parallelize ArUco scale detection and Object Segmentation
            # Cycle time is bounded by max(T_aruco, T_seg) rather than sum!
            # -------------------------------------------------------------
            fut_aruco = self.parallel_executor.submit(
                self.geom_engine.detect_aruco_marker, frame, 50.0
            )
            fut_seg = self.parallel_executor.submit(
                self.segmenter.segment_image, frame
            )

            # Await both concurrent tasks
            found_aruco, corners, ppm, aruco_meta = fut_aruco.result()
            objs = fut_seg.result()

            # Update calibration status
            if found_aruco and ppm is not None:
                calib_status = f"Scale: ArUco 50mm (PPM={ppm:.2f})"
                aruco_info = {"found": True, "corners": corners, "ppm": ppm}
            else:
                calib_status = "Scale: Camera Height Z=500mm (Fixed Plane)"
                aruco_info = {"found": False, "corners": None, "ppm": self.geom_engine.pixels_per_mm or 1.6}

            # Geometry, Feature Extraction & Edge AI Mass Prediction
            results = []
            curr_class = self.current_class

            for obj in objs:
                geom_res = self.geom_engine.compute_metric_dimensions(
                    major_axis_px=obj["major_axis_px"],
                    minor_axis_px=obj["minor_axis_px"],
                    area_px=obj["area_px"],
                    centroid=obj["centroid"],
                    img_shape=frame.shape,
                    object_class=curr_class
                )

                if geom_res["success"]:
                    feat_vec = FeatureExtractor.extract_features(obj, geom_res, frame.shape)
                    pred_out = self.edge_engine.predict_mass(feat_vec)
                    ai_mass = pred_out.get("predicted_mass_g", geom_res["pure_geometric_mass_g"])

                    results.append({
                        "morphology": obj,
                        "geometry": geom_res,
                        "feature_vector": feat_vec,
                        "pure_geometric_mass_g": geom_res["pure_geometric_mass_g"],
                        "ai_predicted_mass_g": ai_mass
                    })

            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0

            # Atomic update of latest results
            with self._results_lock:
                self._latest_results = results
                self._aruco_info = aruco_info
                self._calib_status = calib_status
                self.ai_latency_ms = latency_ms

            ai_frame_count += 1
            self.total_processed_frames += 1
            now = time.time()
            if now - fps_start_time >= 0.5:
                self.ai_fps = ai_frame_count / (now - fps_start_time)
                ai_frame_count = 0
                fps_start_time = now

    def get_render_state(self) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]], Dict[str, Any], str, float, float, float]:
        """
        Fetches the latest fresh camera frame along with the latest AI state snapshot.
        Display delay is bounded strictly by camera frame acquisition rate (shortest process)!
        Returns:
            (frame, results, aruco_info, calib_status, stream_fps, ai_fps, ai_latency_ms)
        """
        frame, _ = self.grabber.get_latest_frame()
        stream_fps = max(1.0, self.grabber.stream_fps)
        fastest_delay_ms = 1000.0 / stream_fps

        with self._results_lock:
            results = list(self._latest_results)
            aruco_info = dict(self._aruco_info)
            calib_status = self._calib_status
            ai_fps = self.ai_fps
            ai_latency_ms = self.ai_latency_ms

        return frame, results, aruco_info, calib_status, stream_fps, ai_fps, ai_latency_ms

    def set_object_class(self, class_name: str):
        if class_name in OBJECT_DENSITIES:
            self.current_class = class_name

    def stop(self):
        self.is_running = False
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=1.0)
        self.parallel_executor.shutdown(wait=False)
        self.grabber.stop()


def draw_industrial_hud(
    frame: np.ndarray,
    results: List[Dict[str, Any]],
    aruco_info: Dict[str, Any],
    object_class: str,
    stream_fps: float,
    ai_fps: float,
    ai_latency_ms: float,
    calib_status: str
) -> np.ndarray:
    """Draws an industrial-grade telemetry HUD overlay on the live camera frame."""
    if frame is None:
        return frame
        
    h, w = frame.shape[:2]
    fastest_delay_ms = 1000.0 / max(1.0, stream_fps)
    
    # 1. Overlay pass for top and bottom control bars
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 58), (15, 23, 42), -1)
    cv2.rectangle(overlay, (0, h - 35), (w, h), (15, 23, 42), -1)

    # Fill segmentation masks / contours into overlay
    for item in results:
        mask = item.get("morphology", {}).get("mask")
        if mask is not None:
            overlay[mask > 0] = (210, 80, 50)
        elif "contour" in item.get("morphology", {}):
            cv2.drawContours(overlay, [item["morphology"]["contour"]], -1, (210, 80, 50), -1)

    # Blend overlay once with 0.70 alpha
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    
    # Top Title & Pipeline Status
    cv2.putText(frame, "INDUSTRIAL DECOUPLED METROLOGY & AI MASS ESTIMATION", (15, 22), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    
    # Primary Telemetry: Stream FPS (Shortest process path) + AI Worker FPS & Latency
    fps_text = f"STREAM: {stream_fps:.1f} FPS ({fastest_delay_ms:.1f}ms) | AI: {ai_fps:.1f} FPS ({ai_latency_ms:.1f}ms)"
    cv2.putText(frame, fps_text, (max(15, w - 500), 22), 
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
        cv2.putText(frame, f"ArUco Scale ({aruco_info.get('ppm', 1.6):.2f} px/mm)", 
                    (corners[0][0], max(20, corners[0][1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw Detected Objects Contours and HUD Telemetry Tags
    for item in results:
        morph = item.get("morphology", {})
        bbox = morph.get("bbox_xywh")
        cnt = morph.get("contour")
        if bbox is None or cnt is None:
            continue
            
        x, y, bw, bh = bbox
        
        # Draw contour & bounding box
        cv2.drawContours(frame, [cnt], -1, (244, 114, 182), 2)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (56, 189, 248), 1)
        
        # Floating telemetry card next to object
        ai_mass = item.get("ai_predicted_mass_g", 0.0)
        geom = item.get("geometry", {})
        
        card_x = min(w - 220, max(10, x))
        card_y = max(65, y - 60)
        
        # Translucent tag card
        cv2.rectangle(frame, (card_x, card_y), (card_x + 210, card_y + 55), (15, 23, 42), -1)
        cv2.rectangle(frame, (card_x, card_y), (card_x + 210, card_y + 55), (56, 189, 248), 1)
        
        mass_txt = f"MASS: {ai_mass:.1f} g"
        cv2.putText(frame, mass_txt, (card_x + 8, card_y + 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (74, 222, 128), 2)
                    
        l_mm = geom.get("length_mm", 0.0)
        w_mm = geom.get("width_mm", 0.0)
        v_cm3 = geom.get("volume_cm3", 0.0)
        dim_txt = f"{l_mm:.0f}x{w_mm:.0f} mm | V:{v_cm3:.0f}cm3"
        cv2.putText(frame, dim_txt, (card_x + 8, card_y + 44), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1)

    return frame
