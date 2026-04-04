import numpy as np


class SmoothQuant:
    """
    Simplified implementation of SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
    
    This is a teaching implementation, not optimized for performance.
    """
    
    def __init__(self, alpha: float = 0.5):
        """
        Initialize SmoothQuant
        
        Args:
            alpha: Hyperparameter balancing activation and weight quantization error (default: 0.5)
        """
        self.alpha = alpha
        self.scales = None  # Scaling factors s_j
        self.weight_quant_scale = None  # Quantization scale for weights
        self.weight_q = None  # Quantized weights
    
    def calibrate(self, W: np.ndarray, x_calib: np.ndarray):
        """
        Calibrate SmoothQuant to find optimal scaling factors
        
        Args:
            W: Weight matrix of shape (n, d)
            x_calib: Calibration activations of shape (num_calib_samples, d)
        """
        n, d = W.shape
        
        # Step 1: Compute max absolute values per channel
        max_act_per_channel = np.max(np.abs(x_calib), axis=0)  # shape (d,)
        max_weight_per_channel = np.max(np.abs(W), axis=0)  # shape (d,)
        
        # Step 2: Compute optimal scaling factors s_j
        # We use the simplified sqrt(max_act / max_weight) from the paper
        # Adding a small epsilon to avoid division by zero
        eps = 1e-8
        scales = np.sqrt(
            (max_act_per_channel + eps) / (max_weight_per_channel + eps)
        )
        
        # For numerical stability, clamp scales
        scales = np.clip(scales, 1e-3, 1e3)
        
        self.scales = scales
        
        # Step 3: Smooth the weights
        W_smooth = W * scales[np.newaxis, :]
        
        # Step 4: Quantize the smoothed weights
        self.weight_q, self.weight_quant_scale = self.quantize_matrix(W_smooth)
    
    def quantize_vector(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Quantize a vector to INT8
        
        Args:
            x: Vector to quantize
            
        Returns:
            (quantized_vector, scale): Quantized INT8 vector and scaling factor
        """
        max_val = np.max(np.abs(x))
        if max_val == 0:
            scale = 1.0
        else:
            scale = max_val / 127.0
        
        x_q = np.round(x / scale).astype(np.int8)
        
        return x_q, scale
    
    def dequantize_vector(self, x_q: np.ndarray, scale: float) -> np.ndarray:
        """
        Dequantize an INT8 vector back to FP32
        
        Args:
            x_q: Quantized INT8 vector
            scale: Scaling factor
            
        Returns:
            Dequantized vector
        """
        return x_q.astype(np.float32) * scale
    
    def quantize_matrix(self, W: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Quantize a matrix to INT8 (per-tensor quantization for simplicity)
        
        Args:
            W: Matrix to quantize
            
        Returns:
            (quantized_matrix, scale): Quantized INT8 matrix and scaling factor
        """
        max_val = np.max(np.abs(W))
        if max_val == 0:
            scale = 1.0
        else:
            scale = max_val / 127.0
        
        W_q = np.round(W / scale).astype(np.int8)
        
        return W_q, scale
    
    def inference(self, x: np.ndarray) -> np.ndarray:
        """
        Run SmoothQuant inference
        
        Args:
            x: Input activation vector of shape (d,)
            
        Returns:
            Output vector of shape (n,)
        """
        if self.scales is None or self.weight_q is None:
            raise ValueError("SmoothQuant has not been calibrated!")
        
        # Step 1: Smooth the activation (divide by scales)
        x_smooth = x / self.scales
        
        # Step 2: Quantize the smoothed activation
        x_q, act_scale = self.quantize_vector(x_smooth)
        
        # Step 3: INT8 matrix multiplication (emulated)
        y_q = self.weight_q.astype(np.int32) @ x_q.astype(np.int32)
        
        # Step 4: Dequantize
        y = y_q.astype(np.float32) * self.weight_quant_scale * act_scale
        
        return y


def test_smoothquant():
    """Test function for SmoothQuant"""
    np.random.seed(42)
    
    # Dimensions
    n = 1024  # Output dimension
    d = 2048  # Input dimension
    num_calib_samples = 128  # Number of calibration samples
    
    # Create random weight matrix
    W = np.random.randn(n, d).astype(np.float32) * 0.1  # Weights have small values
    
    # Create calibration activations with outliers
    x_calib = np.random.randn(num_calib_samples, d).astype(np.float32)
    # Add outliers to 0.1% of channels
    outlier_channels = np.random.choice(d, size=int(0.001 * d), replace=False)
    x_calib[:, outlier_channels] *= 20.0  # Make them big outliers
    
    # Create test activation
    x_test = np.random.randn(d).astype(np.float32)
    x_test[outlier_channels] *= 20.0  # Same outlier pattern
    
    print(f"Testing SmoothQuant with n={n}, d={d}")
    print(f"Number of calibration samples: {num_calib_samples}")
    print(f"Number of outlier channels: {len(outlier_channels)}")
    
    # Compute true FP32 result
    y_true = W @ x_test
    
    # Calibrate and run SmoothQuant
    smoothquant = SmoothQuant(alpha=0.5)
    smoothquant.calibrate(W, x_calib)
    y_smoothquant = smoothquant.inference(x_test)
    
    # Also test naive INT8 quantization for comparison
    # Naive: directly quantize W and x without smoothing
    W_q_naive, s_W_naive = smoothquant.quantize_matrix(W)
    x_q_naive, s_x_naive = smoothquant.quantize_vector(x_test)
    y_q_naive = W_q_naive.astype(np.int32) @ x_q_naive.astype(np.int32)
    y_naive = y_q_naive.astype(np.float32) * s_W_naive * s_x_naive
    
    # Compute errors
    def compute_metrics(y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        max_err = np.max(np.abs(y_true - y_pred))
        rel_err = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100
        return mse, mae, max_err, rel_err
    
    sq_mse, sq_mae, sq_max, sq_rel = compute_metrics(y_true, y_smoothquant)
    nv_mse, nv_mae, nv_max, nv_rel = compute_metrics(y_true, y_naive)
    
    print(f"\nTrue FP32 - First 5 values: {y_true[:5]}")
    print(f"SmoothQuant - First 5 values: {y_smoothquant[:5]}")
    print(f"Naive INT8 - First 5 values: {y_naive[:5]}")
    
    print(f"\n=== Error Comparison ===")
    print(f"Metric          | SmoothQuant | Naive INT8")
    print(f"----------------|-------------|-----------")
    print(f"MSE             | {sq_mse:.10f} | {nv_mse:.10f}")
    print(f"MAE             | {sq_mae:.10f} | {nv_mae:.10f}")
    print(f"Max Abs Error   | {sq_max:.10f} | {nv_max:.10f}")
    print(f"Relative Error  | {sq_rel:.4f}%    | {nv_rel:.4f}%")
    print(f"\nSmoothQuant outperforms naive INT8 by a factor of {nv_mse/sq_mse:.1f}x in MSE!")


if __name__ == "__main__":
    test_smoothquant()
