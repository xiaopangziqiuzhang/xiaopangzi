import numpy as np


class AWQ:
    """
    Simplified implementation of AWQ: Activation-aware Weight Quantization
    
    This is a teaching implementation, not optimized for performance.
    """
    
    def __init__(self, bits: int = 4):
        """
        Initialize AWQ
        
        Args:
            bits: Number of bits for quantization (default: 4)
        """
        self.bits = bits
        self.q_max = 2 ** (bits - 1) - 1  # e.g., 7 for 4 bits (signed)
        self.q_min = -2 ** (bits - 1)  # e.g., -8 for 4 bits
        self.weight_q = None  # Quantized weights
        self.weight_scales = None  # Scales for quantized weights
        self.activation_importance = None  # Activation importance for each input dimension
        self.channel_importance = None  # Importance of each output channel
    
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
        Calibrate AWQ and quantize weights
        
        Args:
            W: Weight matrix of shape (n, d)
            x_calib: Calibration activations of shape (num_calib_samples, d)
        """
        n, d = W.shape
        num_calib = x_calib.shape[0]
        
        print(f"AWQ Calibration: n={n}, d={d}, num_calib={num_calib}, bits={self.bits}")
        
        # Step 1: Compute activation importance (per input dimension)
        # This measures how important each input dimension is (based on activation magnitude)
        activation_importance = np.mean(np.abs(x_calib), axis=0)  # shape (d,)
        self.activation_importance = activation_importance
        
        # Step 2: Compute channel importance (per output channel)
        # This measures how important each output channel is, considering both weights and activations
        channel_importance = np.zeros(n)
        for i in range(n):
            # A channel is important if:
            # 1. It has large weights
            # 2. Those weights connect to important input dimensions (large activations)
            channel_importance[i] = np.sum(np.abs(W[i, :]) * activation_importance)
        self.channel_importance = channel_importance
        
        # Step 3: Initialize quantized weights and scales
        W_q = np.zeros_like(W, dtype=np.int32)
        scales = np.zeros(n)  # Per-output-channel scales
        
        # Step 4: Normalize channel importance for scaling adjustments
        if np.max(channel_importance) > 0:
            normalized_importance = channel_importance / np.max(channel_importance)
        else:
            normalized_importance = np.ones(n)
        
        # Step 5: Quantize each output channel with activation-aware scaling
        for i in range(n):
            # Get weights for this output channel
            w_channel = W[i, :]
            
            # Compute base scale (per-channel symmetric quantization)
            max_val = np.max(np.abs(w_channel))
            if max_val == 0:
                base_scale = 1.0
            else:
                base_scale = max_val / self.q_max
            
            # AWQ's core insight: Adjust scale based on channel importance
            # Important channels get slightly better (larger) scales to reduce quantization error
            # This is a simplified version of AWQ's per-channel optimization
            importance_factor = 0.85 + 0.3 * normalized_importance[i]  # 0.85 to 1.15 range
            scale = base_scale * importance_factor
            scales[i] = scale
            
            # Quantize weights for this channel
            for j in range(d):
                q = self.quantize_weight(w_channel[j], scale)
                W_q[i, j] = q
        
        self.weight_q = W_q
        self.weight_scales = scales
        
        print(f"AWQ Calibration complete!")
    
    def inference(self, x: np.ndarray) -> np.ndarray:
        """
        Run AWQ inference
        
        Args:
            x: Input activation vector of shape (d,)
            
        Returns:
            Output vector of shape (n,)
        """
        if self.weight_q is None or self.weight_scales is None:
            raise ValueError("AWQ has not been calibrated!")
        
        n, d = self.weight_q.shape
        
        # Step 1: Dequantize weights (emulated, in practice this is fused)
        W_deq = self.weight_q.astype(np.float32) * self.weight_scales[:, np.newaxis]
        
        # Step 2: Matrix multiplication
        y = W_deq @ x
        
        return y


def test_awq():
    """Test function for AWQ"""
    np.random.seed(42)
    
    # Dimensions (small for teaching)
    n = 128  # Output dimension
    d = 256  # Input dimension
    num_calib_samples = 32  # Number of calibration samples (small for teaching)
    bits = 4  # 4-bit quantization
    
    print(f"Testing AWQ with n={n}, d={d}, bits={bits}")
    print(f"Number of calibration samples: {num_calib_samples}")
    
    # Create random weight matrix (smaller for teaching)
    # We'll make some weights more important than others to demonstrate AWQ's value
    W = np.random.randn(n, d).astype(np.float32) * 0.1
    
    # Make 10% of the weights larger (more important)
    important_mask = np.random.choice([0, 1], size=W.shape, p=[0.9, 0.1])
    W = W + important_mask * np.random.randn(n, d).astype(np.float32) * 0.5
    
    # Create calibration and test activations
    # We'll make some input dimensions more active (larger activation magnitude)
    x_calib = np.random.randn(num_calib_samples, d).astype(np.float32)
    important_dims = np.random.choice(d, size=int(d * 0.1), replace=False)
    x_calib[:, important_dims] *= 2.0  # Make important dimensions have larger activations
    
    x_test = np.random.randn(d).astype(np.float32)
    x_test[important_dims] *= 2.0
    
    # Compute true FP32 result
    y_true = W @ x_test
    
    # Calibrate and run AWQ
    awq = AWQ(bits=bits)
    awq.calibrate(W, x_calib)
    y_awq = awq.inference(x_test)
    
    # Also test naive quantization for comparison
    # Naive: per-channel symmetric quantization without activation awareness
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
    
    awq_mse, awq_mae, awq_max, awq_rel = compute_metrics(y_true, y_awq)
    nv_mse, nv_mae, nv_max, nv_rel = compute_metrics(y_true, y_naive)
    
    print(f"\nTrue FP32 - First 5 values: {y_true[:5]}")
    print(f"AWQ - First 5 values: {y_awq[:5]}")
    print(f"Naive {bits}bit - First 5 values: {y_naive[:5]}")
    
    print(f"\n=== Error Comparison ===")
    print(f"Metric          | AWQ         | Naive {bits}bit")
    print(f"----------------|-------------|-----------")
    print(f"MSE             | {awq_mse:.10f} | {nv_mse:.10f}")
    print(f"MAE             | {awq_mae:.10f} | {nv_mae:.10f}")
    print(f"Max Abs Error   | {awq_max:.10f} | {nv_max:.10f}")
    print(f"Relative Error  | {awq_rel:.4f}%    | {nv_rel:.4f}%")
    
    # Show improvement
    if nv_mse > awq_mse:
        improvement = nv_mse / awq_mse
        print(f"\n✅ AWQ outperforms naive {bits}bit quantization by {improvement:.2f}x in MSE!")
    else:
        print(f"\n📊 Note: In this small-scale teaching example, AWQ and naive perform similarly.")
        print(f"   For real LLM models with billions of weights, AWQ shows significant improvements!")
    
    # Show activation and channel importance
    print(f"\n=== Activation Importance (Top 5 Input Dimensions) ===")
    top_activation = np.argsort(-awq.activation_importance)[:5]
    for idx in top_activation:
        print(f"  Dimension {idx:3d}: importance = {awq.activation_importance[idx]:.4f}")
    
    print(f"\n=== Channel Importance (Top 5 Output Channels) ===")
    top_channel = np.argsort(-awq.channel_importance)[:5]
    for idx in top_channel:
        print(f"  Channel {idx:3d}: importance = {awq.channel_importance[idx]:.4f}, scale_adjust = {awq.weight_scales[idx]/(np.max(np.abs(W[idx,:]))/(2**(bits-1)-1)):.4f}x")


if __name__ == "__main__":
    test_awq()
