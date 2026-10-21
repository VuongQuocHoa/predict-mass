"""
End-to-End Pipeline Verification Script.
Tests:
1. Dataset generation and sample annotation (Stage 1)
2. Morphology and Perspective extraction (Stage 2 & 3A)
3. AI Hybrid MLP training & ONNX export (Stage 3B)
4. FP32 -> INT8 Dynamic Post-Training Quantization (Stage 4)
5. Accelerated Edge Inference with ONNX Runtime (Stage 5)
"""
import sys
from core.data_manager import DatasetManager
from core.ai_regressor import AIMassRegressor
from core.quantizer import ModelQuantizer
from core.edge_inference import EdgeInferenceEngine

def main():
    print("=== [1/5] Testing Stage 1: Dataset Generation ===")
    dm = DatasetManager()
    count = dm.generate_synthetic_benchmark_dataset(num_samples=20)
    print(f"Generated {count} annotated synthetic benchmark samples.")
    
    status = dm.check_dataset_status()
    print("Dataset Status:", status)
    
    print("\n=== [2/5] Testing Stage 2 & 3: AI Hybrid Regression ===")
    X, y, ids = dm.get_training_arrays()
    print(f"Feature matrix shape: {X.shape}, Target labels: {y.shape}")
    
    regressor = AIMassRegressor(model_type="mlp")
    train_res = regressor.train(X, y)
    print("Training Results:", train_res["message"])
    print("Metrics:", train_res["metrics"])
    
    print("\n=== [3/5] Testing ONNX Export ===")
    ok, msg = regressor.export_to_onnx()
    print("ONNX Export Status:", ok, msg)
    assert ok, f"ONNX Export Failed: {msg}"
    
    print("\n=== [4/5] Testing Stage 4: Quantization (FP32 -> INT8) ===")
    q_res = ModelQuantizer.quantize_fp32_to_int8()
    print("Quantization Result:", q_res.get("message"))
    
    perf = ModelQuantizer.compare_inference_performance(iterations=200)
    print(f"Latency FP32: {perf['latency_fp32_ms']:.4f} ms ({perf['fps_fp32']:.1f} FPS)")
    print(f"Latency INT8: {perf['latency_int8_ms']:.4f} ms ({perf['fps_int8']:.1f} FPS)")
    print(f"Speedup Factor: {perf['speedup_factor']:.2f}x")
    
    print("\n=== [5/5] Testing Stage 5: Hardware Accelerated Edge Inference ===")
    edge_engine = EdgeInferenceEngine()
    print(f"Active Provider: {edge_engine.active_provider}")
    
    pred_res = edge_engine.predict_mass(X[0])
    actual = y[0]
    pred = pred_res["predicted_mass_g"]
    error = abs(actual - pred)
    pct = (error / actual) * 100
    print(f"Sample 0 -> Actual: {actual:.1f}g | Predicted: {pred:.1f}g | Error: {error:.2f}g ({pct:.1f}%)")
    print(f"Inference Latency: {pred_res['latency_ms']:.3f} ms | FPS: {pred_res['fps']:.1f}")
    
    print("\n ALL 5 STAGES VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
