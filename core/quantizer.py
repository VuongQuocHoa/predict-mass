"""
Stage 4: Model Quantization Engine (Post-Training Quantization PTQ: FP32 -> INT8).
Quantizes models to INT8 to reduce memory footprint, power consumption,
and memory bandwidth requirements on Edge AI hardware.
"""
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from config import MODEL_CONFIGS

class ModelQuantizer:
    @staticmethod
    def quantize_fp32_to_int8(input_model_path: Optional[Path] = None, 
                              output_model_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Executes Post-Training Quantization (PTQ) from FP32 to INT8.
        """
        src = input_model_path or MODEL_CONFIGS["mlp_onnx_fp32"]
        dst = output_model_path or MODEL_CONFIGS["mlp_onnx_int8"]
        
        if not Path(src).exists():
            return {
                "success": False,
                "needs_user_input": True,
                "message": f"Không tìm thấy mô hình gốc FP32 tại '{src}'. Vui lòng huấn luyện mô hình ở Giai đoạn 3 và xuất ONNX trước."
            }
            
        try:
            start_t = time.time()
            quantize_dynamic(
                model_input=str(src),
                model_output=str(dst),
                weight_type=QuantType.QInt8
            )
            quant_duration = time.time() - start_t
            
            # Compare file sizes
            size_fp32 = os.path.getsize(src)
            size_int8 = os.path.getsize(dst)
            reduction_pct = ((size_fp32 - size_int8) / max(1, size_fp32)) * 100.0
            
            return {
                "success": True,
                "input_model": str(src),
                "output_model": str(dst),
                "size_fp32_bytes": size_fp32,
                "size_int8_bytes": size_int8,
                "size_fp32_kb": size_fp32 / 1024.0,
                "size_int8_kb": size_int8 / 1024.0,
                "compression_ratio": float(size_fp32 / max(1, size_int8)),
                "reduction_percentage": float(reduction_pct),
                "quantization_time_s": float(quant_duration),
                "message": f"Lượng tử hóa FP32 -> INT8 thành công! Dung lượng giảm {reduction_pct:.1f}% ({size_fp32/1024:.1f}KB -> {size_int8/1024:.1f}KB)."
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Lỗi trong quá trình lượng tử hóa: {str(e)}"
            }

    @staticmethod
    def compare_inference_performance(fp32_path: Optional[Path] = None, 
                                      int8_path: Optional[Path] = None, 
                                      iterations: int = 500) -> Dict[str, Any]:
        """
        Benchmarks latency and throughput between FP32 and INT8 models using ONNX Runtime.
        """
        src_fp32 = fp32_path or MODEL_CONFIGS["mlp_onnx_fp32"]
        src_int8 = int8_path or MODEL_CONFIGS["mlp_onnx_int8"]
        
        if not Path(src_fp32).exists() or not Path(src_int8).exists():
            return {
                "success": False,
                "message": "Cần có cả 2 tệp mô hình FP32 và INT8 để so sánh hiệu năng."
            }
            
        try:
            session_fp32 = ort.InferenceSession(str(src_fp32), providers=["CPUExecutionProvider"])
            session_int8 = ort.InferenceSession(str(src_int8), providers=["CPUExecutionProvider"])
            
            input_name_fp32 = session_fp32.get_inputs()[0].name
            input_name_int8 = session_int8.get_inputs()[0].name
            
            dummy_input = np.random.randn(1, 12).astype(np.float32)
            
            # Warmup
            for _ in range(20):
                session_fp32.run(None, {input_name_fp32: dummy_input})
                session_int8.run(None, {input_name_int8: dummy_input})
                
            # Benchmark FP32
            t0 = time.perf_counter()
            for _ in range(iterations):
                session_fp32.run(None, {input_name_fp32: dummy_input})
            lat_fp32 = ((time.perf_counter() - t0) / iterations) * 1000.0 # ms
            
            # Benchmark INT8
            t1 = time.perf_counter()
            for _ in range(iterations):
                session_int8.run(None, {input_name_int8: dummy_input})
            lat_int8 = ((time.perf_counter() - t1) / iterations) * 1000.0 # ms
            
            speedup = lat_fp32 / max(1e-4, lat_int8)
            del session_fp32, session_int8 # Free sessions
            
            return {
                "success": True,
                "iterations": iterations,
                "latency_fp32_ms": float(lat_fp32),
                "fps_fp32": float(1000.0 / max(1e-4, lat_fp32)),
                "latency_int8_ms": float(lat_int8),
                "fps_int8": float(1000.0 / max(1e-4, lat_int8)),
                "speedup_factor": float(speedup)
            }
        except Exception as e:
            return {"success": False, "message": f"Lỗi so sánh hiệu năng: {str(e)}"}
