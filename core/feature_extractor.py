"""
Feature Extractor: Extracts multi-dimensional perspective and geometric feature vector
to feed into the Machine Learning / MLP Regressor for non-linear error compensation.
"""
import numpy as np
from typing import Dict, Any, List

FEATURE_NAMES = [
    "area_px",
    "major_axis_px",
    "minor_axis_px",
    "aspect_ratio",
    "solidity",
    "perimeter_px",
    "dist_to_center_ratio",
    "perspective_angle_factor",
    "real_length_mm",
    "real_width_mm",
    "geom_volume_cm3",
    "density"
]

class FeatureExtractor:
    @staticmethod
    def extract_features(morph_dict: Dict[str, Any], geom_dict: Dict[str, Any], img_shape: tuple) -> np.ndarray:
        """
        Extracts 12-dimensional feature vector combining 2D mask morphology and 3D perspective geometry.
        Returns: 1D numpy array of float32.
        """
        h_img, w_img = img_shape[:2]
        diag = np.sqrt(h_img**2 + w_img**2)
        
        area_px = morph_dict.get("area_px", 0.0)
        major_axis_px = morph_dict.get("major_axis_px", 0.0)
        minor_axis_px = morph_dict.get("minor_axis_px", 0.0)
        aspect_ratio = morph_dict.get("aspect_ratio", 1.0)
        solidity = morph_dict.get("solidity", 1.0)
        perimeter_px = morph_dict.get("perimeter_px", 0.0)
        
        dist_from_optical_center_px = geom_dict.get("dist_from_optical_center_px", 0.0)
        dist_to_center_ratio = dist_from_optical_center_px / max(1.0, diag)
        
        perspective_angle_factor = geom_dict.get("perspective_angle_factor", 1.0)
        real_length_mm = geom_dict.get("length_mm", 0.0)
        real_width_mm = geom_dict.get("width_mm", 0.0)
        geom_volume_cm3 = geom_dict.get("volume_cm3", 0.0)
        density = geom_dict.get("density", 1.0)
        
        vec = np.array([
            area_px,
            major_axis_px,
            minor_axis_px,
            aspect_ratio,
            solidity,
            perimeter_px,
            dist_to_center_ratio,
            perspective_angle_factor,
            real_length_mm,
            real_width_mm,
            geom_volume_cm3,
            density
        ], dtype=np.float32)
        
        return vec

    @staticmethod
    def get_feature_names() -> List[str]:
        return list(FEATURE_NAMES)
