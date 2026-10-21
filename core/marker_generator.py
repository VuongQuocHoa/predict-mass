"""
Utility to generate printable reference markers (ArUco markers)
that users can print or display on their smartphone/screen for real-time calibration.
"""
import cv2
from pathlib import Path
from config import BASE_DIR

MARKERS_DIR = BASE_DIR / "data" / "reference_markers"
MARKERS_DIR.mkdir(parents=True, exist_ok=True)

def generate_aruco_markers():
    """Generates 50x50mm, 70x70mm, and 100x100mm printable ArUco markers."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # ID 0: 50mm marker (size 400x400 px high-res printable)
    marker_50 = cv2.aruco.generateImageMarker(aruco_dict, 0, 400)
    canvas_50 = cv2.copyMakeBorder(marker_50, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)
    cv2.putText(canvas_50, "ArUco DICT_4X4_50 - ID: 0 (Scale: 50mm x 50mm)", (30, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    path_50 = MARKERS_DIR / "aruco_id0_50mm.png"
    cv2.imwrite(str(path_50), canvas_50)
    
    # ID 1: 70mm marker
    marker_70 = cv2.aruco.generateImageMarker(aruco_dict, 1, 500)
    canvas_70 = cv2.copyMakeBorder(marker_70, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)
    cv2.putText(canvas_70, "ArUco DICT_4X4_50 - ID: 1 (Scale: 70mm x 70mm)", (30, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    path_70 = MARKERS_DIR / "aruco_id1_70mm.png"
    cv2.imwrite(str(path_70), canvas_70)

    return {
        "aruco_50mm": str(path_50),
        "aruco_70mm": str(path_70)
    }

if __name__ == "__main__":
    res = generate_aruco_markers()
    print("Generated printable reference markers:", res)
