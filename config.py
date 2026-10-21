"""
Configuration module for Edge AI Object Mass Prediction System.
Handles camera intrinsics, reference markers, fruit densities, and model parameters.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path

# Root directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

for folder in [DATA_DIR, SAMPLES_DIR, ANNOTATIONS_DIR, MODELS_DIR, STATIC_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Standard densities for fruits and common objects (g/cm^3)
OBJECT_DENSITIES = {
    "apple": {"name": "Táo (Apple)", "density": 0.85, "shape_factor": 0.5236},      # Sphere pi/6 ~ 0.5236
    "orange": {"name": "Cam (Orange)", "density": 0.90, "shape_factor": 0.5236},    # Spheroid
    "banana": {"name": "Chuối (Banana)", "density": 0.95, "shape_factor": 0.4200},  # Curved cylinder
    "mango": {"name": "Xoài (Mango)", "density": 0.92, "shape_factor": 0.5100},     # Ellipsoid
    "lemon": {"name": "Chanh (Lemon)", "density": 0.98, "shape_factor": 0.5200},    # Prolate spheroid
    "tomato": {"name": "Cà chua (Tomato)", "density": 0.91, "shape_factor": 0.5236},# Sphere
    "custom": {"name": "Vật thể tùy chỉnh", "density": 1.00, "shape_factor": 0.5236}
}

# Reference Marker Specifications (e.g. ArUco Marker or Standard Card)
REFERENCE_MARKERS = {
    "aruco_50mm": {
        "name": "ArUco Marker (DICT_4X4_50, 50x50 mm)",
        "width_mm": 50.0,
        "height_mm": 50.0,
        "type": "aruco",
        "dictionary": "DICT_4X4_50"
    },
    "credit_card": {
        "name": "Thẻ ngân hàng / Thẻ căn cước (ID/Credit Card: 85.6 x 53.98 mm)",
        "width_mm": 85.6,
        "height_mm": 53.98,
        "type": "rectangle"
    },
    "coin_standard": {
        "name": "Đồng xu chuẩn (Đường kính 25.0 mm)",
        "width_mm": 25.0,
        "height_mm": 25.0,
        "type": "circle"
    },
    "calibrated_plane": {
        "name": "Mặt bàn phẳng cố định khoảng cách Z (Fixed Camera Height)",
        "width_mm": None,
        "height_mm": None,
        "type": "fixed_z"
    }
}

# Default Camera Intrinsics (Standard 720p / 1080p Webcams)
DEFAULT_CAMERA_PARAMS = {
    "focal_length_px": 800.0,    # Focal length fx, fy in pixels
    "default_distance_z_mm": 500.0,  # 50cm from camera to table
    "image_width": 1280,
    "image_height": 720
}

# Models configurations
MODEL_CONFIGS = {
    "yolo_nano_weights": MODELS_DIR / "yolo11n-seg.pt",
    "mlp_onnx_fp32": MODELS_DIR / "mass_regressor_mlp.onnx",
    "mlp_onnx_int8": MODELS_DIR / "mass_regressor_int8.onnx",
    "dataset_json": DATA_DIR / "dataset.json"
}

# Hardware Acceleration Providers Priority
EXECUTION_PROVIDERS_PRIORITY = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "OpenVINOExecutionProvider",
    "CPUExecutionProvider"
]
