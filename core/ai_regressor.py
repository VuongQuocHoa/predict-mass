"""
Stage 3B: Machine Learning & MLP Neural Network Regressor.
Combines geometric features with non-linear deep learning/ML to compensate
for perspective distortion, irregular organic fruit shapes, and shadow artifacts,
accurately predicting physical mass (grams).
Exports model directly to standard ONNX format for Edge AI deployment.
"""
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from config import MODEL_CONFIGS

class AIMassRegressor:
    def __init__(self, model_type: str = "mlp"):
        self.model_type = model_type
        self.pipeline: Optional[Pipeline] = None
        self.is_trained = False
        self.metrics = {}
        self.feature_names = [
            "area_px", "major_axis_px", "minor_axis_px", "aspect_ratio",
            "solidity", "perimeter_px", "dist_to_center_ratio", "perspective_angle_factor",
            "real_length_mm", "real_width_mm", "geom_volume_cm3", "density"
        ]

    def build_pipeline(self, n_samples: int = 50) -> Pipeline:
        scaler = StandardScaler()
        if self.model_type == "mlp":
            # For edge datasets (< 500 samples), L-BFGS converges with machine precision
            # whereas Adam requires thousands of batches.
            solver_choice = "lbfgs" if n_samples < 500 else "adam"
            model = MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver=solver_choice,
                alpha=0.05,
                max_iter=600,
                random_state=42
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                random_state=42
            )
        return Pipeline([("scaler", scaler), ("regressor", model)])

    def train(self, X: np.ndarray, y: np.ndarray, val_split: float = 0.2) -> Dict[str, Any]:
        """
        Trains the AI hybrid regression model and computes MAE, RMSE, R2.
        """
        if len(X) < 5:
            return {
                "success": False,
                "needs_user_input": True,
                "message": f"Dữ liệu huấn luyện quá ít ({len(X)} mẫu). Cần tối thiểu 5 mẫu gán nhãn khối lượng để huấn luyện AI."
            }

        num_samples = len(X)
        val_size = max(1, int(num_samples * val_split))
        indices = np.random.RandomState(42).permutation(num_samples)
        
        train_idx, val_idx = indices[val_size:], indices[:val_size]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        self.pipeline = self.build_pipeline(n_samples=len(X_train))
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        
        # Predictions
        y_pred_val = self.pipeline.predict(X_val)
        y_pred_val = np.maximum(y_pred_val, 1.0) # mass must be > 0
        
        mae = float(mean_absolute_error(y_val, y_pred_val))
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
        r2 = float(r2_score(y_val, y_pred_val)) if len(y_val) > 1 else 1.0
        pct_err = float(np.mean(np.abs((y_val - y_pred_val) / np.maximum(y_val, 1.0))) * 100)
        
        self.metrics = {
            "samples_count": num_samples,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "mae_grams": mae,
            "rmse_grams": rmse,
            "r2_score": max(-1.0, r2),
            "mean_percentage_error": pct_err
        }
        
        # Save local joblib checkpoint
        checkpoint_path = MODEL_CONFIGS["mlp_onnx_fp32"].with_suffix(".joblib")
        joblib.dump(self.pipeline, checkpoint_path)
        
        return {
            "success": True,
            "metrics": self.metrics,
            "message": f"Huấn luyện thành công! MAE sai số trung bình: {mae:.2f}g ({pct_err:.1f}%)"
        }

    def predict(self, feature_vec: np.ndarray) -> float:
        """
        Predicts physical mass (g) for a single feature vector or batch.
        """
        if not self.is_trained or self.pipeline is None:
            # Fallback to pure geometric mass (feature index 10 is volume, index 11 is density)
            if feature_vec.ndim == 1:
                return float(feature_vec[10] * feature_vec[11])
            else:
                return float(feature_vec[0, 10] * feature_vec[0, 11])
                
        if feature_vec.ndim == 1:
            feature_vec = feature_vec.reshape(1, -1)
            
        pred = self.pipeline.predict(feature_vec)
        return float(max(1.0, pred[0]))

    def export_to_onnx(self, output_path: Optional[Path] = None) -> Tuple[bool, str]:
        """
        Exports the trained pipeline (StandardScaler + Regressor) to standard ONNX.
        """
        if not self.is_trained or self.pipeline is None:
            return False, "Mô hình chưa được huấn luyện. Vui lòng huấn luyện trước khi xuất ONNX."
            
        target_path = output_path or MODEL_CONFIGS["mlp_onnx_fp32"]
        
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            
            initial_type = [("float_input", FloatTensorType([None, len(self.feature_names)]))]
            onx = convert_sklearn(self.pipeline, initial_types=initial_type, target_opset=14)
            
            with open(target_path, "wb") as f:
                f.write(onx.SerializeToString())
                
            file_size_kb = target_path.stat().st_size / 1024.0
            return True, f"Xuất ONNX thành công: {target_path.name} ({file_size_kb:.1f} KB)"
        except Exception as e:
            return False, f"Lỗi khi chuyển đổi ONNX: {str(e)}"

    def load_from_checkpoint(self, path: Optional[Path] = None) -> bool:
        """Loads trained pipeline from disk if available."""
        target_path = path or MODEL_CONFIGS["mlp_onnx_fp32"].with_suffix(".joblib")
        if target_path.exists():
            try:
                self.pipeline = joblib.load(target_path)
                self.is_trained = True
                return True
            except Exception:
                return False
        return False
