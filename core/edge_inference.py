"""
Stage 5: Hardware Acceleration & Edge Inference Engine.
Leverages ONNX Runtime with execution providers:
- OpenVINOExecutionProvider (Intel CPU / iGPU / NPU)
- TensorrtExecutionProvider / CUDAExecutionProvider (NVIDIA GPU)
- CPUExecutionProvider (Universal Fallback)
Provides low-latency edge inference and real-time hardware telemetry.
"""
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import onnxruntime as ort
from config import MODEL_CONFIGS, EXECUTION_PROVIDERS_PRIORITY

class EdgeInferenceEngine:
    def __init__(self, model_path: Optional[Path] = None, preferred_provider: Optional[str] = None):
        self.model_path = model_path or MODEL_CONFIGS["mlp_onnx_int8"]
        # If INT8 model doesn't exist yet, fallback to FP32 ONNX
        if not Path(self.model_path).exists() and MODEL_CONFIGS["mlp_onnx_fp32"].exists():
            self.model_path = MODEL_CONFIGS["mlp_onnx_fp32"]
            
        self.available_providers = ort.get_available_providers()
        self.active_provider = "None"
        self.session = None
        self.input_name = None
        self.output_name = None
        
        self._init_session(preferred_provider)

    def _init_session(self, preferred_provider: Optional[str] = None):
        """Initializes ONNX Runtime InferenceSession with chosen or best hardware provider."""
        if not Path(self.model_path).exists():
            return
            
        # Determine candidate providers
        selected_providers = []
        if preferred_provider and preferred_provider in self.available_providers:
            selected_providers.append(preferred_provider)
            
        for ep in EXECUTION_PROVIDERS_PRIORITY:
            if ep in self.available_providers and ep not in selected_providers:
                selected_providers.append(ep)
                
        # Always add CPUExecutionProvider at end as safe fallback
        if "CPUExecutionProvider" not in selected_providers:
            selected_providers.append("CPUExecutionProvider")
            
        try:
            # Session options for edge performance
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            self.session = ort.InferenceSession(
                str(self.model_path), 
                sess_options=opts, 
                providers=selected_providers
            )
            self.active_provider = self.session.get_providers()[0]
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        except Exception as e:
            # Fallback strictly to CPU if specialized provider init encounters driver issue
            self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
            self.active_provider = "CPUExecutionProvider"
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

    def predict_mass(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """
        Runs accelerated ONNX inference on the input feature vector.
        """
        if self.session is None:
            return {
                "success": False,
                "needs_user_input": True,
                "message": "Chưa có tệp mô hình ONNX để suy luận. Vui lòng huấn luyện và xuất ONNX trước."
            }
            
        if feature_vector.ndim == 1:
            inp = feature_vector.reshape(1, -1).astype(np.float32)
        else:
            inp = feature_vector.astype(np.float32)
            
        start_t = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: inp})
        latency_ms = (time.perf_counter() - start_t) * 1000.0
        
        predicted_mass = float(outputs[0][0][0]) if outputs[0].ndim > 1 else float(outputs[0][0])
        predicted_mass = max(1.0, predicted_mass)
        
        return {
            "success": True,
            "predicted_mass_g": predicted_mass,
            "latency_ms": latency_ms,
            "fps": float(1000.0 / max(1e-4, latency_ms)),
            "active_provider": self.active_provider,
            "model_path": str(self.model_path)
        }

    def benchmark_providers(self, iterations: int = 300) -> Dict[str, Any]:
        """
        Benchmarks inference speed across all available execution providers on this machine.
        """
        if not Path(self.model_path).exists():
            return {"success": False, "message": "Chưa tìm thấy tệp mô hình ONNX."}
            
        dummy_input = np.random.randn(1, 12).astype(np.float32)
        results = {}
        
        for prov in self.available_providers:
            try:
                sess = ort.InferenceSession(str(self.model_path), providers=[prov])
                inp_name = sess.get_inputs()[0].name
                # Warmup
                for _ in range(15):
                    sess.run(None, {inp_name: dummy_input})
                    
                t0 = time.perf_counter()
                for _ in range(iterations):
                    sess.run(None, {inp_name: dummy_input})
                avg_ms = ((time.perf_counter() - t0) / iterations) * 1000.0
                del sess # Free ONNX session memory
                
                results[prov] = {
                    "latency_ms": float(avg_ms),
                    "fps": float(1000.0 / max(1e-4, avg_ms)),
                    "status": "Operational"
                }
            except Exception as e:
                results[prov] = {
                    "latency_ms": None,
                    "fps": None,
                    "status": f"Unsupported or Missing Driver ({str(e)[:40]})"
                }
                
        return {
            "success": True,
            "benchmarks": results,
            "active_provider": self.active_provider,
            "available_providers": self.available_providers
        }
