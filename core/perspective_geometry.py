"""
Stage 3A: Perspective Depth Geometry & Calibration Engine.
Calculates real-world dimensions (mm), 3D volume (cm^3), and baseline geometric mass (g)
using perspective projection, camera intrinsics, and reference markers (ArUco or Reference Card).
"""
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
from config import REFERENCE_MARKERS, DEFAULT_CAMERA_PARAMS, OBJECT_DENSITIES

class PerspectiveGeometryEngine:
    def __init__(self, focal_length_px: float = 800.0, default_distance_z: float = 500.0):
        self.focal_length_px = focal_length_px
        self.distance_z_mm = default_distance_z
        self.pixels_per_mm = None
        self.smooth_ppm = None
        self.calibration_source = "None"
        
        # Initialize ArUco detector using modern OpenCV API with optimized parameters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 23
        self.aruco_params.adaptiveThreshWinSizeStep = 10
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def detect_aruco_marker(self, image: np.ndarray, marker_real_size_mm: float = 50.0) -> Tuple[bool, Optional[np.ndarray], Optional[float], Dict[str, Any]]:
        """
        Detects ArUco marker in the image and calculates pixels-per-millimeter (PPM).
        Optimized to release rejected candidates and prevent memory accumulation.
        Returns: (found, corners, ppm, metadata)
        """
        if image is None or image.size == 0:
            return False, None, None, {}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        del rejected # Free memory immediately from hundreds of candidate shapes
        
        if ids is not None and len(ids) > 0:
            # Use the first detected marker
            c = corners[0].reshape(4, 2) # guarantee shape (4, 2)
            marker_id = int(np.ravel(ids)[0])
            # Edge lengths in pixels
            top_edge = float(np.linalg.norm(c[0] - c[1]))
            right_edge = float(np.linalg.norm(c[1] - c[2]))
            bottom_edge = float(np.linalg.norm(c[2] - c[3]))
            left_edge = float(np.linalg.norm(c[3] - c[0]))
            avg_edge_px = (top_edge + right_edge + bottom_edge + left_edge) / 4.0
            
            raw_ppm = avg_edge_px / max(1.0, marker_real_size_mm)
            if self.smooth_ppm is None:
                self.smooth_ppm = raw_ppm
            else:
                self.smooth_ppm = 0.85 * self.smooth_ppm + 0.15 * raw_ppm
                
            ppm = float(self.smooth_ppm)
            self.pixels_per_mm = ppm
            self.calibration_source = f"ArUco Marker ID {marker_id} ({marker_real_size_mm}mm)"
            
            # Perspective distortion: ratio of opposite sides
            h_ratio = min(top_edge, bottom_edge) / max(1e-6, max(top_edge, bottom_edge))
            v_ratio = min(left_edge, right_edge) / max(1e-6, max(left_edge, right_edge))
            tilt_factor = (h_ratio + v_ratio) / 2.0
            
            return True, c, ppm, {
                "id": marker_id,
                "avg_edge_px": float(avg_edge_px),
                "ppm": float(ppm),
                "tilt_factor": float(tilt_factor)
            }
        
        return False, None, None, {}

    def calibrate_with_card(self, image: np.ndarray, card_real_w: float = 85.6, card_real_h: float = 53.98) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        """
        Detects a rectangular reference card (e.g. credit/ID card).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(cnt) > 2000:
                rect = cv2.minAreaRect(cnt)
                w, h = rect[1]
                major_px = max(w, h)
                minor_px = min(w, h)
                # Check aspect ratio (~1.585 for credit card)
                ratio = major_px / (minor_px + 1e-6)
                if 1.4 < ratio < 1.8:
                    ppm = major_px / card_real_w
                    self.pixels_per_mm = ppm
                    self.calibration_source = f"Standard Reference Card ({card_real_w}x{card_real_h}mm)"
                    return True, approx, ppm
        return False, None, None

    def calibrate_with_distance(self, distance_z_mm: float, focal_length_px: Optional[float] = None) -> float:
        """
        Perspective depth calibration from fixed camera distance Z and focal length f.
        PPM = f / Z
        """
        f = focal_length_px if focal_length_px is not None else self.focal_length_px
        self.distance_z_mm = distance_z_mm
        self.pixels_per_mm = f / max(1.0, distance_z_mm)
        self.calibration_source = f"Fixed Camera Distance Z={distance_z_mm}mm (f={f:.1f}px)"
        return self.pixels_per_mm

    def compute_metric_dimensions(self, 
                                  major_axis_px: float, 
                                  minor_axis_px: float, 
                                  area_px: float, 
                                  centroid: Tuple[float, float],
                                  img_shape: Tuple[int, int],
                                  object_class: str = "apple") -> Dict[str, Any]:
        """
        Converts 2D pixel measurements to 3D metric dimensions and estimates volume & mass.
        Returns detailed measurements, or indicates what missing data is needed.
        """
        if self.pixels_per_mm is None or self.pixels_per_mm <= 0:
            return {
                "success": False,
                "needs_user_input": True,
                "error_code": "MISSING_CALIBRATION",
                "message": "Hệ thống chưa có thước đo tỷ lệ (PPM). Vui lòng đặt ArUco Marker vào ảnh hoặc nhập Khoảng cách Camera Z (mm)."
            }

        # 1. Physical Length and Width in mm
        length_mm = major_axis_px / self.pixels_per_mm
        width_mm = minor_axis_px / self.pixels_per_mm
        
        # 2. Perspective offset correction based on centroid position relative to optical center
        h_img, w_img = img_shape[:2]
        cx, cy = centroid
        center_x, center_y = w_img / 2.0, h_img / 2.0
        delta_x = cx - center_x
        delta_y = cy - center_y
        dist_from_optical_center = np.sqrt(delta_x**2 + delta_y**2)
        
        # Radial distortion / perspective angle factor cos(theta)
        # theta is angle between optical axis and ray to object
        tan_theta = dist_from_optical_center / self.focal_length_px
        cos_theta = 1.0 / np.sqrt(1.0 + tan_theta**2)
        perspective_scale_factor = float(cos_theta)
        
        # Correct dimensions for perspective angle
        corrected_length_mm = length_mm / max(0.5, cos_theta)
        corrected_width_mm = width_mm / max(0.5, cos_theta)
        
        # 3. Estimate Physical Height (Z dimension) based on object class shape factor
        obj_info = OBJECT_DENSITIES.get(object_class, OBJECT_DENSITIES["apple"])
        shape_factor = obj_info["shape_factor"]
        density = obj_info["density"]
        
        # For typical fruits, height is closely correlated with minor axis
        # (e.g. apple is oblate spheroid: Height ~ Minor diameter)
        estimated_height_mm = corrected_width_mm
        if object_class == "banana":
            estimated_height_mm = corrected_width_mm * 0.95
        elif object_class == "lemon":
            estimated_height_mm = corrected_width_mm * 1.05
            
        # 4. Geometric Volume Calculation (cm^3)
        # Spheroid Volume = (4/3) * pi * (L/2) * (W/2) * (H/2) = (pi/6) * L * W * H
        volume_mm3 = shape_factor * (corrected_length_mm * corrected_width_mm * estimated_height_mm)
        volume_cm3 = volume_mm3 / 1000.0  # 1 cm^3 = 1000 mm^3
        
        # 5. Pure Geometric Mass (grams) = Volume (cm^3) * Density (g/cm^3)
        geom_mass_g = volume_cm3 * density
        
        return {
            "success": True,
            "needs_user_input": False,
            "pixels_per_mm": float(self.pixels_per_mm),
            "calibration_source": self.calibration_source,
            "length_mm": float(corrected_length_mm),
            "width_mm": float(corrected_width_mm),
            "height_mm": float(estimated_height_mm),
            "volume_cm3": float(volume_cm3),
            "density": float(density),
            "pure_geometric_mass_g": float(geom_mass_g),
            "perspective_angle_factor": float(perspective_scale_factor),
            "dist_from_optical_center_px": float(dist_from_optical_center)
        }
