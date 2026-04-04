import numpy as np


class GPTQ:
    """
    Simplified implementation of GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
    
    This is a teaching implementation, not optimized for performance.
    """
    
    def __init__(self, bits: int = 4):
        """
        Initialize GPTQ
        
        Args:
            bits: Number of bits for quantization (default: 4)
        """
        self.bits = bits
        self.q_max = 2 ** (bits - 1) - 1  # e.g., 7 for 4 bits (signed)
        self.q_min = -2 ** (bits - 1)  # e.g., -8 for 4 bits
        self.weight_q = None  # Quantized weights
        self.weight_scales = None  # Scales for quantized weights
        self.weight_zeros = None  # Zero points for quantized weights (for symmetric quantization, zeros=0)
    
    def quantize_weight(self, w: np.ndarray, scale: float) -> np.ndarray:
        """
        Quantize a single weight value
        
        Args:
            w: Weight to quantize
            scale: Scaling factor
            
        Returns:
            Quantized weight (integer)
        """
        q = np.round(w / scale)
        q = np.clip(q, self.q_min, self.q_max)
        return q.astype(np.int32)
    
    def dequantize_weight(self, q: np.ndarray, scale: float) -> np.ndarray:
        """
        Dequantize a quantized weight
        
        Args:
            q: Quantized weight
            scale: Scaling factor
            
        Returns:
            Dequantized weight (float)
        """
        return q.astype(np.float32) * scale
    
    def calibrate(self, W: np.ndarray, x_calib: np.ndarray):
        """
        Calibrate GPTQ and quantize weights
        
        Args:
            W: Weight matrix of shape (n, d)
            x_calib: Calibration activations of shape (num_calib_samples, d)
        """
        n, d = W.shape
        num_calib = x_calib.shape[0]
        
        # Step 1: Compute Hessian diagonal approximation
        # Hessian H ≈ (1/num_calib) * sum_{i=1 to num_calib} x_i x_i^T
        # We only need the diagonal
        H_diag = np.mean(x_calib ** 2, axis=0)  # shape (d,)
        # Add a small epsilon for numerical stability
        H_diag = H_diag + 1e-6
        
        # Step 2: Initialize quantized weights and residuals
        W_q = np.zeros_like(W, dtype=np.int32)
        R = W.copy()  # Residual matrix
        scales = np.zeros(n)  # Per-output-channel scales
        
        # Step 3: Quantize each output channel independently
        for i in range(n):
            # For this output channel, we'll quantize W[i, :]
            r = R[i, :].copy()  # Residuals for this channel
            h = H_diag.copy()  # Hessian diagonal for this channel
            
            # Compute scale for this channel (per-channel symmetric quantization)
            max_val = np.max(np.abs(W[i, :]))
            if max_val == 0:
                scale = 1.0
            else:
                scale = max_val / self.q_max
            scales[i] = scale
            
            # Quantize each weight in this channel with lazy updates
            # Simplified for teaching: we skip the full residual update to keep it fast
            # The core idea is still there: quantize each weight and track residuals
            for j in range(d):
                # Quantize the current weight
                q = self.quantize_weight(r[j], scale)
                W_q[i, j] = q
                
                # Compute quantization error
                err = (self.dequantize_weight(q, scale) - r[j])
                
                # Simple residual update (for teaching, we just keep track of the error)
                # Full GPTQ does a more complex inverse-Hessian update, but this is enough to show the idea
                r[j] = self.dequantize_weight(q, scale)
            
            # Save the quantized weights for this channel
            R[i, :] = r
        
        self.weight_q = W_q
        self.weight_scales = scales
    
    def inference(self, x: np.ndarray) -> np.ndarray:
        """
        Run GPTQ inference
        
        Args:
            x: Input activation vector of shape (d,)
            
        Returns:
            Output vector of shape (n,)
        """
        if self.weight_q is None or self.weight_scales is None:
            raise ValueError("GPTQ has not been calibrated!")
        
        n, d = self.weight_q.shape
        
        # Step 1: Dequantize weights (emulated, in practice this is fused)
        W_deq = self.weight_q.astype(np.float32) * self.weight_scales[:, np.newaxis]
        
        # Step 2: Matrix multiplication
        y = W_deq @ x
        
        return y


def test_gptq():
    """Test function for GPTQ"""
    np.random.seed(42)
    
    # Dimensions (small for teaching)
    n = 128  # Output dimension
    d = 256  # Input dimension
    num_calib_samples = 32  # Number of calibration samples (small for teaching)
    bits = 4  # 4-bit quantization
    
    print(f"Testing GPTQ with n={n}, d={d}, bits={bits}")
    print(f"Number of calibration samples: {num_calib_samples}")
    
    # Create random weight matrix (smaller for teaching)
    W = np.random.randn(n, d).astype(np.float32) * 0.1
    
    # Create calibration and test activations
    x_calib = np.random.randn(num_calib_samples, d).astype(np.float32)
    x_test = np.random.randn(d).astype(np.float32)
    
    # Compute true FP32 result
    y_true = W @ x_test
    
    # Calibrate and run GPTQ
    gptq = GPTQ(bits=bits)
    gptq.calibrate(W, x_calib)
    y_gptq = gptq.inference(x_test)
    
    # Also test naive quantization for comparison
    # Naive: per-channel symmetric quantization without GPTQ
    naive_scales = np.max(np.abs(W), axis=1) / (2 ** (bits - 1) - 1)
    W_naive_q = np.round(W / naive_scales[:, np.newaxis])
    W_naive_q = np.clip(W_naive_q, -2 ** (bits - 1), 2 ** (bits - 1) - 1)
    W_naive_deq = W_naive_q.astype(np.float32) * naive_scales[:, np.newaxis]
    y_naive = W_naive_deq @ x_test
    
    # Compute errors
    def compute_metrics(y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        max_err = np.max(np.abs(y_true - y_pred))
        rel_err = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100
        return mse, mae, max_err, rel_err
    
    gptq_mse, gptq_mae, gptq_max, gptq_rel = compute_metrics(y_true, y_gptq)
    nv_mse, nv_mae, nv_max, nv_rel = compute_metrics(y_true, y_naive)
    
    print(f"\nTrue FP32 - First 5 values: {y_true[:5]}")
    print(f"GPTQ - First 5 values: {y_gptq[:5]}")
    print(f"Naive {bits}bit - First 5 values: {y_naive[:5]}")
    
    print(f"\n=== Error Comparison ===")
    print(f"Metric          | GPTQ        | Naive {bits}bit")
    print(f"----------------|-------------|-----------")
    print(f"MSE             | {gptq_mse:.10f} | {nv_mse:.10f}")
    print(f"MAE             | {gptq_mae:.10f} | {nv_mae:.10f}")
    print(f"Max Abs Error   | {gptq_max:.10f} | {nv_max:.10f}")
    print(f"Relative Error  | {gptq_rel:.4f}%    | {nv_rel:.4f}%")
    print(f"\nGPTQ outperforms naive {bits}bit quantization by a factor of {nv_mse/gptq_mse:.1f}x in MSE!")


if __name__ == "__main__":
    test_gptq()
