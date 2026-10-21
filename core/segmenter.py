"""
Stage 2: Model Architecture & Image Segmentation Engine (Optimized for Edge AI).
Supports YOLOv11n-seg (Nano) and High-Speed Adaptive Contour Segmenter.
Extracts bounding boxes, segmentation masks, contours, fitted ellipses,
major/minor axes, solidity, and morphological features with minimal RAM overhead.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import gc
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from config import MODEL_CONFIGS

class ObjectSegmenter:
    def __init__(self, use_yolo: bool = True, model_path: Optional[Path] = None, conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        self.use_yolo = use_yolo
        self.yolo_model = None
        self.model_status = "Not Initialized"
        
        if use_yolo:
            self._init_yolo(model_path)
        else:
            self.model_status = "Adaptive Contour Segmentation (Edge Fallback)"

    def _init_yolo(self, model_path: Optional[Path] = None):
        """Attempts to load YOLOv11n-seg model."""
        try:
            from ultralytics import YOLO
            target_path = str(model_path or MODEL_CONFIGS["yolo_nano_weights"])
            if not Path(target_path).exists():
                self.yolo_model = YOLO("yolo11n-seg.pt")
            else:
                self.yolo_model = YOLO(target_path)
            self.model_status = "YOLOv11n-seg (Ultralytics Nano Model Ready)"
        except Exception as e:
            self.yolo_model = None
            self.model_status = f"YOLO Load Warning ({str(e)[:40]}). Using Adaptive Segmenter."

    def segment_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Segments objects in image and returns detailed morphological characteristics.
        Limits peak resolution to prevent RAM explosion on high-res camera frames.
        """
        if image is None or image.size == 0:
            return []

        # Safe downscaling for excessively large images (e.g., 4K phone uploads)
        h, w = image.shape[:2]
        max_dim = max(h, w)
        if max_dim > 1280:
            scale = 1280.0 / max_dim
            proc_img = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            proc_img = image

        if self.yolo_model is not None:
            try:
                res = self._segment_with_yolo(proc_img)
            except Exception:
                res = self._segment_adaptive(proc_img)
        else:
            res = self._segment_adaptive(proc_img)

        # Rescale results back to original image coordinates if image was downscaled
        if scale != 1.0 and res:
            inv_scale = 1.0 / scale
            for obj in res:
                obj["area_px"] *= (inv_scale ** 2)
                obj["perimeter_px"] *= inv_scale
                obj["major_axis_px"] *= inv_scale
                obj["minor_axis_px"] *= inv_scale
                cx, cy = obj["centroid"]
                obj["centroid"] = (cx * inv_scale, cy * inv_scale)
                x, y, bw, bh = obj["bbox_xywh"]
                obj["bbox_xywh"] = (int(x * inv_scale), int(y * inv_scale), int(bw * inv_scale), int(bh * inv_scale))
                if "contour" in obj and obj["contour"] is not None:
                    obj["contour"] = (obj["contour"] * inv_scale).astype(np.int32)
                if "mask" in obj and obj["mask"] is not None:
                    obj["mask"] = cv2.resize(obj["mask"], (w, h), interpolation=cv2.INTER_NEAREST)

        return res

    def _segment_with_yolo(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Runs YOLOv11n-seg inference with memory-safe inference mode and explicit cleanup."""
        results = None
        try:
            import torch
            with torch.inference_mode():
                results = self.yolo_model(image, conf=self.conf_threshold, imgsz=320, verbose=False)
        except Exception:
            return self._segment_adaptive(image)
            
        detected_objects = []
        if results is None or len(results) == 0:
            return detected_objects
            
        res = results[0]
        h_orig, w_orig = image.shape[:2]
        
        # Check if masks exist
        if res.masks is not None and len(res.masks) > 0:
            boxes = res.boxes
            masks_data = res.masks.data.cpu().numpy() # shape (N, H, W)
            
            for i, mask in enumerate(masks_data):
                mask_uint8 = (mask * 255).astype(np.uint8)
                if mask_uint8.shape[:2] != (h_orig, w_orig):
                    mask_uint8 = cv2.resize(mask_uint8, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                
                cls_id = int(boxes.cls[i].item())
                cls_name = self.yolo_model.names.get(cls_id, f"obj_{cls_id}")
                conf = float(boxes.conf[i].item())
                
                # Extract contour & morphology directly
                morph = self._analyze_mask_morphology(mask_uint8, image.shape)
                if morph["area_px"] > 400: # Filter out tiny noise
                    morph["class_id"] = cls_id
                    morph["class_name"] = cls_name
                    morph["confidence"] = conf
                    morph["mask"] = mask_uint8
                    detected_objects.append(morph)
                    
        # Fallback if YOLO detected bounding boxes but no segmentation masks
        elif res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                cls_id = int(box.cls[0].item())
                cls_name = self.yolo_model.names.get(cls_id, f"obj_{cls_id}")
                conf = float(box.conf[0].item())
                
                roi = image[max(0, y1):min(h_orig, y2), max(0, x1):min(w_orig, x2)]
                if roi.size > 0:
                    sub_mask = self._segment_roi(roi)
                    full_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
                    full_mask[max(0, y1):min(h_orig, y2), max(0, x1):min(w_orig, x2)] = sub_mask
                    
                    morph = self._analyze_mask_morphology(full_mask, image.shape)
                    if morph["area_px"] > 400:
                        morph["class_id"] = cls_id
                        morph["class_name"] = cls_name
                        morph["confidence"] = conf
                        morph["mask"] = full_mask
                        detected_objects.append(morph)

        # Free PyTorch / Ultralytics result objects
        del results, res
        
        # If YOLO found nothing, attempt adaptive edge fallback
        if len(detected_objects) == 0:
            detected_objects = self._segment_adaptive(image)
            
        return detected_objects

    def _segment_adaptive(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        High-speed adaptive segmentation using Otsu thresholding and direct contour analysis.
        Avoids redundant full-mask allocations and duplicate findContours calls.
        """
        h_orig, w_orig = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_objects = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter noise and background canvas borders
            if 500 < area < (h_orig * w_orig * 0.9):
                morph = self._analyze_contour_morphology(cnt, image.shape)
                morph["class_id"] = 0
                morph["class_name"] = "detected_object"
                morph["confidence"] = 0.90
                
                # Create mask on-demand
                mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                morph["mask"] = mask
                
                detected_objects.append(morph)
                
        return detected_objects

    def _segment_roi(self, roi: np.ndarray) -> np.ndarray:
        """Segments a local bounding box region."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh

    def _analyze_contour_morphology(self, main_contour: np.ndarray, img_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """
        Directly calculates morphological metrics from contour points with ZERO array reallocation.
        """
        area_px = float(cv2.contourArea(main_contour))
        perimeter_px = float(cv2.arcLength(main_contour, True))
        
        # Centroid from spatial moments
        M = cv2.moments(main_contour)
        if M["m00"] != 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            cx, cy = 0.0, 0.0
            
        # Bounding box
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Minimum area rotated rectangle
        rect = cv2.minAreaRect(main_contour)
        (center_rect), (rect_w, rect_h), angle = rect
        major_axis_px = float(max(rect_w, rect_h))
        minor_axis_px = float(min(rect_w, rect_h))
        
        # Fitted ellipse (requires at least 5 points)
        if len(main_contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(main_contour)
                (e_cx, e_cy), (e_w, e_h), e_angle = ellipse
                major_axis_px = float(max(e_w, e_h))
                minor_axis_px = float(min(e_w, e_h))
            except Exception:
                pass
                
        # Convex hull and solidity
        hull = cv2.convexHull(main_contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = float(area_px / max(1.0, hull_area))
        
        # Aspect ratio
        aspect_ratio = float(major_axis_px / max(1.0, minor_axis_px))
        
        return {
            "area_px": area_px,
            "perimeter_px": perimeter_px,
            "major_axis_px": major_axis_px,
            "minor_axis_px": minor_axis_px,
            "aspect_ratio": aspect_ratio,
            "solidity": solidity,
            "centroid": (cx, cy),
            "bbox_xywh": (int(x), int(y), int(w), int(h)),
            "contour": main_contour,
            "hull": hull
        }

    def _analyze_mask_morphology(self, mask: np.ndarray, img_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """
        Extracts rich morphological and perspective features from binary mask.
        Kept for backward compatibility and mask-first pipelines.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"area_px": 0.0, "perimeter_px": 0.0, "major_axis_px": 0.0, "minor_axis_px": 0.0,
                    "aspect_ratio": 1.0, "solidity": 1.0, "centroid": (0.0, 0.0),
                    "bbox_xywh": (0, 0, 0, 0), "contour": None, "hull": None}
            
        main_contour = max(contours, key=cv2.contourArea)
        return self._analyze_contour_morphology(main_contour, img_shape)
