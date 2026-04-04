import numpy as np


class LLMInt8:
    """
    Simplified implementation of LLM.int8: 8-bit Matrix Multiplication for Transformers at Scale
    
    This is a teaching implementation, not optimized for performance.
    The real bitsandbytes library uses CUDA kernels and is much faster.
    """
    
    def __init__(self, outlier_threshold: float = 6.0):
        """
        Initialize LLM.int8 quantizer
        
        Args:
            outlier_threshold: Threshold for detecting outliers (default: 6.0)
        """
        self.outlier_threshold = outlier_threshold
    
    def detect_outlier_channels(self, x: np.ndarray) -> np.ndarray:
        """
        Detect which channels contain outliers
        
        Args:
            x: Activation vector of shape (d,) or batch of vectors (batch_size, d)
            
        Returns:
            Boolean array of shape (d,) indicating outlier channels
        """
        if x.ndim == 1:
            x = x[np.newaxis, :]
        
        # Check which channels have any value exceeding the threshold
        outlier_channels = np.max(np.abs(x), axis=0) > self.outlier_threshold
        
        return outlier_channels
    
    def quantize_vector(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Quantize a vector to INT8
        
        Args:
            x: Vector to quantize
            
        Returns:
            (quantized_vector, scale): Quantized INT8 vector and scaling factor
        """
        # Compute scale to map x's range to INT8 range [-127, 127]
        max_val = np.max(np.abs(x))
        if max_val == 0:
            scale = 1.0
        else:
            scale = max_val / 127.0
        
        # Quantize
        x_q = np.round(x / scale).astype(np.int8)
        
        return x_q, scale
    
    def dequantize_vector(self, x_q: np.ndarray, scale: float) -> np.ndarray:
        """
        Dequantize an INT8 vector back to FP16/FP32
        
        Args:
            x_q: Quantized INT8 vector
            scale: Scaling factor used for quantization
            
        Returns:
            Dequantized vector
        """
        return x_q.astype(np.float32) * scale
    
    def matmul(self, W: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Perform LLM.int8 mixed-precision matrix multiplication y = Wx
        
        Args:
            W: Weight matrix of shape (n, d)
            x: Activation vector of shape (d,)
            
        Returns:
            Output vector of shape (n,)
        """
        # Step 1: Detect outlier channels
        outlier_channels = self.detect_outlier_channels(x)
        
        # Step 2: Split into outlier and non-outlier parts
        # Outlier part (FP16)
        x_out = x.copy()
        x_out[~outlier_channels] = 0.0  # Zero out non-outlier channels
        W_out = W.copy()
        W_out[:, ~outlier_channels] = 0.0  # Zero out non-outlier columns
        
        # Non-outlier part (INT8)
        x_in = x.copy()
        x_in[outlier_channels] = 0.0  # Zero out outlier channels
        W_in = W.copy()
        W_in[:, outlier_channels] = 0.0  # Zero out outlier columns
        
        # Step 3: Compute outlier part in FP16/FP32
        y_out = W_out @ x_out
        
        # Step 4: Compute non-outlier part in INT8
        if np.any(~outlier_channels):  # Only if there are non-outlier channels
            # Quantize weights (per-tensor quantization for simplicity)
            W_in_q, s_W = self.quantize_vector(W_in.flatten())
            W_in_q = W_in_q.reshape(W_in.shape)
            
            # Quantize activation
            x_in_q, s_x = self.quantize_vector(x_in)
            
            # INT8 matrix multiplication (emulated with FP32 for this demo)
            y_in_q = W_in_q.astype(np.int32) @ x_in_q.astype(np.int32)
            
            # Dequantize
            y_in = y_in_q.astype(np.float32) * s_W * s_x
        else:
            y_in = 0.0
        
        # Step 5: Combine both parts
        y = y_out + y_in
        
        return y


def test_llm_int8():
    """Test function for LLM.int8"""
    np.random.seed(42)
    
    # Dimensions
    n = 1024  # Output dimension
    d = 2048  # Input dimension
    
    # Create random weight matrix
    W = np.random.randn(n, d).astype(np.float32)
    
    # Create activation vector with some outliers
    x = np.random.randn(d).astype(np.float32)
    # Add outliers to 0.1% of channels
    outlier_indices = np.random.choice(d, size=int(0.001 * d), replace=False)
    x[outlier_indices] *= 20.0  # Make them big outliers
    
    print(f"Testing LLM.int8 with n={n}, d={d}")
    print(f"Number of outlier channels: {len(outlier_indices)}")
    
    # Compute true FP32 result
    y_true = W @ x
    
    # Compute LLM.int8 result
    llm_int8 = LLMInt8(outlier_threshold=6.0)
    y_llm_int8 = llm_int8.matmul(W, x)
    
    # Compute error
    mse = np.mean((y_true - y_llm_int8) ** 2)
    mae = np.mean(np.abs(y_true - y_llm_int8))
    max_error = np.max(np.abs(y_true - y_llm_int8))
    
    print(f"\nTrue FP32 - First 5 values: {y_true[:5]}")
    print(f"LLM.int8 - First 5 values: {y_llm_int8[:5]}")
    print(f"\nError metrics:")
    print(f"  MSE: {mse:.10f}")
    print(f"  MAE: {mae:.10f}")
    print(f"  Max absolute error: {max_error:.10f}")
    print(f"  Relative error (mean): {np.mean(np.abs((y_true - y_llm_int8) / (np.abs(y_true) + 1e-10))):.6f}%")


if __name__ == "__main__":
    test_llm_int8()
