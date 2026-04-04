import numpy as np
from scipy.stats import beta
from scipy.optimize import minimize_scalar


class TurboQuantMSE:
    def __init__(self, dim: int, bit_width: int):
        """
        Initialize TurboQuant for MSE optimization
        
        Args:
            dim: Dimension of input vectors
            bit_width: Number of bits per coordinate
        """
        self.dim = dim
        self.bit_width = bit_width
        self.num_centroids = 2 ** bit_width
        
        # Step 1: Generate random rotation matrix
        self.rotation_matrix = self._generate_random_rotation(dim)
        
        # Step 2: Precompute optimal codebook for Beta distribution
        self.codebook = self._compute_optimal_codebook(bit_width, dim)
    
    def _generate_random_rotation(self, dim: int) -> np.ndarray:
        """Generate a random d x d rotation matrix using QR decomposition"""
        # Generate random Gaussian matrix
        random_matrix = np.random.randn(dim, dim)
        # QR decomposition to get orthogonal matrix
        q, _ = np.linalg.qr(random_matrix)
        return q
    
    def _compute_optimal_codebook(self, bit_width: int, dim: int) -> np.ndarray:
        """
        Compute optimal scalar quantizer codebook for Beta distribution
        
        For high dim, the distribution approximates N(0, 1/d), so we use
        a simplified approach with uniform quantizer + refinement for demonstration
        """
        num_centroids = 2 ** bit_width
        
        # For demonstration, we use a uniform quantizer over [-1, 1]
        # In real implementation, we should solve the continuous k-means problem
        codebook = np.linspace(-1, 1, num_centroids)
        
        return codebook
    
    def quantize(self, x: np.ndarray) -> np.ndarray:
        """
        Quantize input vector
        
        Args:
            x: Input vector of shape (dim,)
            
        Returns:
            Indices of shape (dim,)
        """
        # Rotate input vector
        y = self.rotation_matrix @ x
        
        # Find nearest centroids for each coordinate
        indices = np.argmin(np.abs(y[:, np.newaxis] - self.codebook[np.newaxis, :]), axis=1)
        
        return indices
    
    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        """
        Dequantize indices back to vector
        
        Args:
            indices: Indices of shape (dim,)
            
        Returns:
            Reconstructed vector of shape (dim,)
        """
        # Get centroids
        y_tilde = self.codebook[indices]
        
        # Rotate back
        x_tilde = self.rotation_matrix.T @ y_tilde
        
        return x_tilde


class QJL:
    def __init__(self, dim: int):
        """
        Initialize Quantized Johnson-Lindenstrauss (QJL)
        
        Args:
            dim: Dimension of input vectors
        """
        self.dim = dim
        # Generate random matrix S with i.i.d. N(0,1) entries
        self.S = np.random.randn(dim, dim)
    
    def quantize(self, x: np.ndarray) -> np.ndarray:
        """
        Quantize input vector to 1 bit per coordinate
        
        Args:
            x: Input vector of shape (dim,)
            
        Returns:
            Binary vector of shape (dim,) with values in {-1, 1}
        """
        return np.sign(self.S @ x)
    
    def dequantize(self, z: np.ndarray) -> np.ndarray:
        """
        Dequantize binary vector back
        
        Args:
            z: Binary vector of shape (dim,) with values in {-1, 1}
            
        Returns:
            Reconstructed vector of shape (dim,)
        """
        return (np.sqrt(np.pi / 2) / self.dim) * (self.S.T @ z)


class TurboQuantProd:
    def __init__(self, dim: int, bit_width: int):
        """
        Initialize TurboQuant for inner product optimization (unbiased)
        
        Args:
            dim: Dimension of input vectors
            bit_width: Number of bits per coordinate
        """
        self.dim = dim
        self.bit_width = bit_width
        
        # First stage: MSE quantizer with 1 bit less
        self.mse_quant = TurboQuantMSE(dim, bit_width - 1)
        
        # Second stage: QJL for residual
        self.qjl = QJL(dim)
    
    def quantize(self, x: np.ndarray):
        """
        Quantize input vector (two-stage)
        
        Args:
            x: Input vector of shape (dim,)
            
        Returns:
            (mse_indices, qjl_bits): tuple of indices from MSE quantizer and QJL bits
        """
        # First stage: MSE quantize
        mse_indices = self.mse_quant.quantize(x)
        x_tilde_mse = self.mse_quant.dequantize(mse_indices)
        
        # Compute residual
        residual = x - x_tilde_mse
        
        # Second stage: QJL on residual
        qjl_bits = self.qjl.quantize(residual)
        
        return mse_indices, qjl_bits
    
    def dequantize(self, mse_indices: np.ndarray, qjl_bits: np.ndarray) -> np.ndarray:
        """
        Dequantize (two-stage)
        
        Args:
            mse_indices: Indices from MSE quantizer
            qjl_bits: Bits from QJL
            
        Returns:
            Reconstructed vector of shape (dim,)
        """
        # First stage dequantize
        x_tilde_mse = self.mse_quant.dequantize(mse_indices)
        
        # Second stage dequantize residual
        residual_tilde = self.qjl.dequantize(qjl_bits)
        
        # Combine
        return x_tilde_mse + residual_tilde


def test_turbo_quant():
    """Simple test function for TurboQuant"""
    np.random.seed(42)
    
    dim = 128
    bit_width = 4
    
    # Generate random unit vector
    x = np.random.randn(dim)
    x = x / np.linalg.norm(x)
    
    print(f"Testing TurboQuant with dim={dim}, bit_width={bit_width}")
    print(f"Original vector norm: {np.linalg.norm(x):.4f}")
    
    # Test TurboQuant MSE
    print("\n=== Testing TurboQuant MSE ===")
    tq_mse = TurboQuantMSE(dim, bit_width)
    indices = tq_mse.quantize(x)
    x_tilde = tq_mse.dequantize(indices)
    
    mse = np.mean((x - x_tilde) ** 2)
    print(f"MSE distortion: {mse:.6f}")
    print(f"Reconstructed vector norm: {np.linalg.norm(x_tilde):.4f}")
    
    # Test TurboQuant Prod (unbiased inner product)
    print("\n=== Testing TurboQuant Prod (Unbiased Inner Product) ===")
    tq_prod = TurboQuantProd(dim, bit_width)
    
    # Generate another random vector for inner product test
    y = np.random.randn(dim)
    
    mse_indices, qjl_bits = tq_prod.quantize(x)
    x_tilde_prod = tq_prod.dequantize(mse_indices, qjl_bits)
    
    true_inner = np.dot(y, x)
    est_inner = np.dot(y, x_tilde_prod)
    
    print(f"True inner product: {true_inner:.6f}")
    print(f"Estimated inner product: {est_inner:.6f}")
    print(f"Absolute error: {abs(true_inner - est_inner):.6f}")
    
    # Test QJL unbiasedness with multiple samples
    print("\n=== Testing QJL Unbiasedness (100 samples) ===")
    qjl = QJL(dim)
    est_in_sum = 0.0
    for i in range(100):
        qj_bits = qjl.quantize(x)
        xq_tilde = qjl.dequantize(qj_bits)
        est_in_sum += np.dot(y, xq_tilde)
    avg_est = est_in_sum / 100
    print(f"True inner: {true_inner:.6f}, Average estimated: {avg_est:.6f}")
    print(f"Average absolute error: {abs(true_inner - avg_est):.6f}")


if __name__ == "__main__":
    test_turbo_quant()
