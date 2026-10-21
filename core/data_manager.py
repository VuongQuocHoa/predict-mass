"""
Stage 1: Dataset & Annotation Management (CVAT / Roboflow Format).
Handles image storage, polygon segmentation masks, ground-truth weights,
data validation, and missing information alerting.
Includes realistic sample fruit dataset generator with ArUco reference markers.
"""
import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from config import DATA_DIR, SAMPLES_DIR, ANNOTATIONS_DIR, MODEL_CONFIGS, OBJECT_DENSITIES

class DatasetManager:
    def __init__(self, registry_file: Optional[Path] = None):
        self.registry_file = registry_file or MODEL_CONFIGS["dataset_json"]
        self.dataset: List[Dict[str, Any]] = []
        self.load_dataset()

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Loads dataset items from JSON registry."""
        if Path(self.registry_file).exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    self.dataset = json.load(f)
            except Exception:
                self.dataset = []
        else:
            self.dataset = []
        return self.dataset

    def save_dataset(self):
        """Saves dataset to JSON registry."""
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, indent=2, ensure_ascii=False)

    def add_sample(self, 
                   image_path: str, 
                   object_class: str, 
                   feature_vector: np.ndarray, 
                   geom_info: Dict[str, Any], 
                   actual_weight_g: Optional[float] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Adds a labeled sample. If actual_weight_g is missing, flags as needing user input.
        """
        needs_input = actual_weight_g is None or actual_weight_g <= 0
        
        sample_id = f"sample_{len(self.dataset) + 1:04d}"
        item = {
            "id": sample_id,
            "image_path": str(image_path),
            "object_class": object_class,
            "actual_weight_g": actual_weight_g if not needs_input else None,
            "pure_geometric_mass_g": geom_info.get("pure_geometric_mass_g", 0.0),
            "length_mm": geom_info.get("length_mm", 0.0),
            "width_mm": geom_info.get("width_mm", 0.0),
            "volume_cm3": geom_info.get("volume_cm3", 0.0),
            "density": geom_info.get("density", 1.0),
            "pixels_per_mm": geom_info.get("pixels_per_mm", None),
            "calibration_source": geom_info.get("calibration_source", "None"),
            "features": feature_vector.tolist(),
            "needs_user_input": needs_input,
            "metadata": metadata or {}
        }
        
        self.dataset.append(item)
        self.save_dataset()
        
        if needs_input:
            return {
                "success": True,
                "sample_id": sample_id,
                "needs_user_input": True,
                "alert": f"⚠️ Mẫu '{sample_id}' ({object_class}) đã được trích xuất hình học nhưng CHƯA CÓ khối lượng thực tế (Ground Truth). Vui lòng cân vật thể và nhập số gram để hệ thống huấn luyện AI."
            }
        return {
            "success": True,
            "sample_id": sample_id,
            "needs_user_input": False,
            "alert": f"Đã lưu mẫu '{sample_id}' ({object_class}) với khối lượng thực tế: {actual_weight_g:.1f}g."
        }

    def update_ground_truth_weight(self, sample_id: str, actual_weight_g: float) -> bool:
        """Updates the ground-truth mass for a specific sample."""
        for item in self.dataset:
            if item["id"] == sample_id:
                item["actual_weight_g"] = float(actual_weight_g)
                item["needs_user_input"] = False
                self.save_dataset()
                return True
        return False

    def get_training_arrays(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Retrieves feature matrix X and ground truth mass array y for all labeled samples.
        """
        X_list = []
        y_list = []
        ids = []
        
        for item in self.dataset:
            if item.get("actual_weight_g") is not None and item.get("features") is not None:
                X_list.append(item["features"])
                y_list.append(item["actual_weight_g"])
                ids.append(item["id"])
                
        if len(X_list) == 0:
            return np.empty((0, 12), dtype=np.float32), np.empty((0,), dtype=np.float32), []
            
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), ids

    def check_dataset_status(self) -> Dict[str, Any]:
        """Checks dataset completeness and identifies missing user inputs."""
        total = len(self.dataset)
        labeled = sum(1 for x in self.dataset if x.get("actual_weight_g") is not None)
        unlabeled = total - labeled
        
        ready_for_training = labeled >= 5
        
        alerts = []
        if total == 0:
            alerts.append("Hệ thống chưa có dữ liệu mẫu. Bạn có thể bấm 'Khởi tạo tập dữ liệu mẫu' để nạp mẫu trái cây chuẩn.")
        elif unlabeled > 0:
            alerts.append(f"Có {unlabeled} mẫu đang thiếu nhãn khối lượng thực tế. Cần nhập khối lượng để bổ sung vào tập huấn luyện AI.")
        elif not ready_for_training:
            alerts.append(f"Hiện có {labeled}/5 mẫu đã gán nhãn. Cần tối thiểu 5 mẫu để bắt đầu huấn luyện mô hình MLP.")
            
        return {
            "total_samples": total,
            "labeled_samples": labeled,
            "unlabeled_samples": unlabeled,
            "ready_for_training": ready_for_training,
            "alerts": alerts
        }

    def generate_synthetic_benchmark_dataset(self, num_samples: int = 25) -> int:
        """
        Generates realistic synthetic fruit test images with rendered ArUco markers
        and realistic physical variations (apples, oranges, bananas, lemons, mangoes)
        to demonstrate the complete end-to-end pipeline instantly.
        """
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, 0, 100) # 100x100 px ArUco marker
        
        # Color palette (BGR) for synthetic fruits
        fruit_specs = [
            {"cls": "apple", "color": (34, 45, 210), "min_r": 35, "max_r": 50, "density": 0.85, "ellipticity": 0.95},
            {"cls": "orange", "color": (20, 140, 245), "min_r": 32, "max_r": 48, "density": 0.90, "ellipticity": 1.00},
            {"cls": "lemon", "color": (30, 215, 235), "min_r": 25, "max_r": 38, "density": 0.98, "ellipticity": 1.35},
            {"cls": "mango", "color": (40, 185, 230), "min_r": 35, "max_r": 55, "density": 0.92, "ellipticity": 1.45},
            {"cls": "banana", "color": (60, 220, 240), "min_r": 20, "max_r": 35, "density": 0.95, "ellipticity": 2.20}
        ]
        
        created = 0
        self.dataset = [] # Reset or populate
        
        for i in range(num_samples):
            spec = fruit_specs[i % len(fruit_specs)]
            cls_name = spec["cls"]
            
            # Create a 640x480 canvas representing a tabletop conveyor/measurement surface
            canvas = np.full((480, 640, 3), 245, dtype=np.uint8) # light gray background
            
            # Subtle grid table texture
            for g_x in range(0, 640, 40):
                cv2.line(canvas, (g_x, 0), (g_x, 480), (235, 235, 235), 1)
            for g_y in range(0, 480, 40):
                cv2.line(canvas, (0, g_y), (640, g_y), (235, 235, 235), 1)
                
            # Place ArUco marker in the top-left corner
            # 100px represents 50mm -> PPM = 2.0 pixels/mm
            m_h, m_w = marker_img.shape
            canvas[30:30+m_h, 30:30+m_w] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(canvas, (28, 28), (32+m_w, 32+m_h), (50, 50, 50), 2)
            cv2.putText(canvas, "ArUco 50mm", (30, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
            
            # Fruit dimensions
            ppm = 2.0 # 2 pixels per mm
            r_minor_px = np.random.uniform(spec["min_r"] * ppm, spec["max_r"] * ppm)
            r_major_px = r_minor_px * spec["ellipticity"] * np.random.uniform(0.95, 1.05)
            
            # Object center placed with slight perspective variation
            center_x = int(np.random.uniform(320, 480))
            center_y = int(np.random.uniform(200, 340))
            angle = float(np.random.uniform(-45, 45))
            
            # Draw shadow
            shadow_center = (center_x + 8, center_y + 10)
            cv2.ellipse(canvas, shadow_center, (int(r_major_px * 0.95), int(r_minor_px * 0.95)), 
                        angle, 0, 360, (210, 210, 210), -1)
                        
            # Draw organic fruit body
            cv2.ellipse(canvas, (center_x, center_y), (int(r_major_px), int(r_minor_px)), 
                        angle, 0, 360, spec["color"], -1)
            # Add highlight for 3D appearance
            highlight_center = (center_x - int(r_major_px * 0.25), center_y - int(r_minor_px * 0.25))
            cv2.ellipse(canvas, highlight_center, (int(r_major_px * 0.4), int(r_minor_px * 0.4)), 
                        angle, 0, 360, (255, 255, 255), -1)
            cv2.GaussianBlur(canvas, (3, 3), 0, dst=canvas)
            
            # Calculate physical dimensions and ground-truth mass with realistic organic density variation
            length_mm = (r_major_px * 2.0) / ppm
            width_mm = (r_minor_px * 2.0) / ppm
            height_mm = width_mm * 0.95
            
            volume_cm3 = (np.pi / 6.0) * (length_mm * width_mm * height_mm) / 1000.0
            # Realistic actual weight with slight natural variation (+- 3%)
            organic_density = spec["density"] * np.random.uniform(0.97, 1.03)
            actual_mass_g = float(volume_cm3 * organic_density)
            pure_geom_mass_g = float(volume_cm3 * spec["density"])
            
            # Save image
            img_file = SAMPLES_DIR / f"{cls_name}_sample_{i+1:02d}.jpg"
            cv2.imwrite(str(img_file), canvas)
            
            # Extract features
            area_px = float(np.pi * r_major_px * r_minor_px)
            perimeter_px = float(np.pi * (3 * (r_major_px + r_minor_px) - np.sqrt((3 * r_major_px + r_minor_px) * (r_major_px + 3 * r_minor_px))))
            dist_to_center = float(np.sqrt((center_x - 320)**2 + (center_y - 240)**2))
            aspect_ratio = float(r_major_px / r_minor_px)
            
            feature_vec = np.array([
                area_px,
                float(r_major_px * 2.0),
                float(r_minor_px * 2.0),
                aspect_ratio,
                0.98, # solidity
                perimeter_px,
                dist_to_center / 800.0,
                0.96, # perspective factor
                length_mm,
                width_mm,
                volume_cm3,
                spec["density"]
            ], dtype=np.float32)
            
            self.add_sample(
                image_path=str(img_file),
                object_class=cls_name,
                feature_vector=feature_vec,
                geom_info={
                    "pure_geometric_mass_g": pure_geom_mass_g,
                    "length_mm": length_mm,
                    "width_mm": width_mm,
                    "volume_cm3": volume_cm3,
                    "density": spec["density"],
                    "pixels_per_mm": ppm,
                    "calibration_source": "ArUco Marker 50mm"
                },
                actual_weight_g=actual_mass_g,
                metadata={"synthetic": True, "natural_variation_pct": float((organic_density/spec["density"] - 1.0)*100)}
            )
            created += 1
            
        return created
