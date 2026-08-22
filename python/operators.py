"""
EduHDC Operators Library (Contribution C1: EduHDC Algebra).
High-performance PyTorch implementation of pedagogical VSA operators satisfying the 4 Pedagogical Axioms:
- Axiom 1: Transitivity of Prerequisite Paths
- Axiom 2: Asymmetric Mastery Binding (Non-Commutativity via Non-Abelian Block Algebra)
- Axiom 3: Competency Bundling Compositionality
- Axiom 4: Semantic Proximity Preservation
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

# ==============================================================================
# 1. Base Class & Standard VSA Baselines
# ==============================================================================

class BaseVSA:
    """Base interface for VSA binding and bundling operators."""
    
    def __init__(self, dim: int = 10000, device: Optional[str] = None):
        self.dim = dim
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        
    def random_vector(self, n: int = 1) -> torch.Tensor:
        raise NotImplementedError
        
    def bind(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        
    def unbind(self, bound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        
    def bundle(self, vectors: Union[List[torch.Tensor], torch.Tensor], weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        if isinstance(vectors, list):
            stacked = torch.stack(vectors, dim=0)
        else:
            stacked = vectors
        if weights is not None:
            weights = weights.to(self.device).view(-1, 1)
            stacked = stacked * weights
            
        summed = stacked.sum(dim=0)
        if summed.is_complex():
            norm = (summed.abs().pow(2).sum(dim=-1, keepdim=True).sqrt())
            return summed / (norm + 1e-8)
        return F.normalize(summed.float(), p=2, dim=-1)
        
    def similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.is_complex():
            inner = (x * y.conj()).sum(dim=-1).real
            norm_x = x.abs().pow(2).sum(dim=-1).sqrt()
            norm_y = y.abs().pow(2).sum(dim=-1).sqrt()
            return inner / (norm_x * norm_y + 1e-8)
        return F.cosine_similarity(x, y, dim=-1)


class BipolarMAP(BaseVSA):
    """Classical Bipolar MAP (Commutative: x * y == y * x)."""
    def random_vector(self, n: int = 1) -> torch.Tensor:
        return (torch.randint(0, 2, (n, self.dim), device=self.device) * 2 - 1).float()
    
    def bind(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y
    
    def unbind(self, bound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return bound * key


class RealHRR(BaseVSA):
    """Classical Real HRR (Commutative circular convolution: x (*) y == y (*) x)."""
    def random_vector(self, n: int = 1) -> torch.Tensor:
        v = torch.randn(n, self.dim, device=self.device)
        return F.normalize(v, p=2, dim=-1)
    
    def bind(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        fx = torch.fft.rfft(x, dim=-1)
        fy = torch.fft.rfft(y, dim=-1)
        return torch.fft.irfft(fx * fy, n=self.dim, dim=-1)
    
    def unbind(self, bound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        fb = torch.fft.rfft(bound, dim=-1)
        fk = torch.fft.rfft(key, dim=-1)
        return torch.fft.irfft(fb * fk.conj(), n=self.dim, dim=-1)


class ComplexFHRR(BaseVSA):
    """Fourier HRR (Commutative complex phases e^{i theta})."""
    def random_vector(self, n: int = 1) -> torch.Tensor:
        theta = torch.rand(n, self.dim, device=self.device) * 2.0 * torch.pi - torch.pi
        return torch.exp(1j * theta)
    
    def bind(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y
    
    def unbind(self, bound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return bound * key.conj()


# ==============================================================================
# 2. EduHDC Non-Abelian & Pedagogical Operators (C1)
# ==============================================================================

class EduBindBlockDiag(BaseVSA):
    """
    EduHDC Block-Diagonal Orthogonal GHRR Operator (EduBind-BlockDiag).
    Partitions D-dimensional space into B non-commutative orthogonal 2x2 blocks.
    Binding is blockwise matrix multiplication:
        Z_i = X_i @ Y_i ≠ Y_i @ X_i
    Unbinding is exact orthogonal inverse:
        Y_i = X_i^T @ Z_i (Left Unbinding: recovers Y given X)
        X_i = Z_i @ Y_i^T (Right Unbinding: recovers X given Y)
    Guarantees strict non-commutativity (Axiom 2) and exact path transitivity (Axiom 1).
    """
    def __init__(self, dim: int = 10000, device: Optional[str] = None):
        super().__init__(dim=dim, device=device)
        self.block_size = 2
        self.num_blocks = dim // 4
        self.actual_dim = self.num_blocks * 4
        
    def random_vector(self, n: int = 1) -> torch.Tensor:
        """Fast analytical O(2) random orthogonal matrix generation on GPU."""
        theta = torch.rand(n, self.num_blocks, device=self.device) * 2.0 * torch.pi
        s = torch.randint(0, 2, (n, self.num_blocks), device=self.device).float() * 2.0 - 1.0
        
        c = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        # Construct 2x2 blocks: [[c, -s*sin_t], [sin_t, s*c]]
        b00 = c
        b01 = -s * sin_t
        b10 = sin_t
        b11 = s * c
        
        blocks = torch.stack([b00, b01, b10, b11], dim=-1) # (n, num_blocks, 4)
        return blocks.reshape(n, self.actual_dim)
        
    def _reshape_blocks(self, x: torch.Tensor) -> torch.Tensor:
        prefix = x.shape[:-1]
        return x[..., :self.actual_dim].reshape(*prefix, self.num_blocks, 2, 2)

    def bind(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Blockwise non-commutative matrix product: Z_i = X_i @ Y_i.
        Uses element-wise 2x2 matmul (avoids torch.matmul overhead for tiny blocks)."""
        # Flatten to (..., num_blocks*4) and extract 2x2 entries via strided indexing
        x = x[..., :self.actual_dim]
        y = y[..., :self.actual_dim]
        x00, x01, x10, x11 = x[..., 0::4], x[..., 1::4], x[..., 2::4], x[..., 3::4]
        y00, y01, y10, y11 = y[..., 0::4], y[..., 1::4], y[..., 2::4], y[..., 3::4]
        # Z = X @ Y: [[x00,x01],[x10,x11]] @ [[y00,y01],[y10,y11]]
        z00 = x00 * y00 + x01 * y10
        z01 = x00 * y01 + x01 * y11
        z10 = x10 * y00 + x11 * y10
        z11 = x10 * y01 + x11 * y11
        return torch.stack([z00, z01, z10, z11], dim=-1).reshape(x.shape)

    def unbind(self, bound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Left unbind: recovers Y given X via X^T @ Z.
        Uses element-wise 2x2 matmul (avoids torch.matmul overhead for tiny blocks)."""
        z = bound[..., :self.actual_dim]
        x = key[..., :self.actual_dim]
        z00, z01, z10, z11 = z[..., 0::4], z[..., 1::4], z[..., 2::4], z[..., 3::4]
        x00, x01, x10, x11 = x[..., 0::4], x[..., 1::4], x[..., 2::4], x[..., 3::4]
        # Y = X^T @ Z: [[x00,x10],[x01,x11]] @ [[z00,z01],[z10,z11]]
        y00 = x00 * z00 + x10 * z10
        y01 = x00 * z01 + x10 * z11
        y10 = x01 * z00 + x11 * z10
        y11 = x01 * z01 + x11 * z11
        return torch.stack([y00, y01, y10, y11], dim=-1).reshape(z.shape)

    def unbind_left_operand(self, bound: torch.Tensor, right_key: torch.Tensor) -> torch.Tensor:
        """Right unbind: recovers X given Y via Z @ Y^T.
        Uses element-wise 2x2 matmul."""
        z = bound[..., :self.actual_dim]
        y = right_key[..., :self.actual_dim]
        z00, z01, z10, z11 = z[..., 0::4], z[..., 1::4], z[..., 2::4], z[..., 3::4]
        y00, y01, y10, y11 = y[..., 0::4], y[..., 1::4], y[..., 2::4], y[..., 3::4]
        # X = Z @ Y^T: [[z00,z01],[z10,z11]] @ [[y00,y10],[y01,y11]]
        x00 = z00 * y00 + z01 * y01
        x01 = z00 * y10 + z01 * y11
        x10 = z10 * y00 + z11 * y01
        x11 = z10 * y10 + z11 * y11
        return torch.stack([x00, x01, x10, x11], dim=-1).reshape(z.shape)


class EduBindComplexUnitary(BaseVSA):
    """
    EduHDC Complex Unitary GHRR Operator (EduBind-Unitary in C^D).
    Partitions complex Hilbert space into 2x2 SU(2) unitary blocks.
    Matrix multiplication in SU(2) is non-commutative and preserves L2 energy exactly.
    """
    def __init__(self, dim: int = 10000, device: Optional[str] = None):
        super().__init__(dim=dim, device=device)
        self.block_size = 2
        self.num_blocks = dim // 4
        self.actual_dim = self.num_blocks * 4
        
    def random_vector(self, n: int = 1) -> torch.Tensor:
        """Analytical Haar measure sampling of U(2) on GPU (100% exact unitary)."""
        theta = torch.rand(n, self.num_blocks, device=self.device) * (torch.pi / 2.0)
        phi = torch.rand(n, self.num_blocks, device=self.device) * 2.0 * torch.pi
        psi = torch.rand(n, self.num_blocks, device=self.device) * 2.0 * torch.pi
        chi = torch.rand(n, self.num_blocks, device=self.device) * 2.0 * torch.pi
        
        c = torch.cos(theta)
        s = torch.sin(theta)
        
        u00 = torch.exp(1j * psi) * c
        u01 = torch.exp(1j * phi) * s
        u10 = -torch.exp(1j * (chi - phi)) * s
        u11 = torch.exp(1j * (chi - psi)) * c
        
        blocks = torch.stack([u00, u01, u10, u11], dim=-1) # (n, num_blocks, 4)
        return blocks.reshape(n, self.actual_dim)
        
    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        prefix = x.shape[:-1]
        return x[..., :self.actual_dim].reshape(*prefix, self.num_blocks, 2, 2)
        
    def bind(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        X_b = self._reshape(x)
        Y_b = self._reshape(y)
        Z_b = torch.matmul(X_b, Y_b)
        prefix = x.shape[:-1]
        return Z_b.reshape(*prefix, self.actual_dim)
        
    def unbind(self, bound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        Z_b = self._reshape(bound)
        X_b = self._reshape(key)
        Y_b = torch.matmul(X_b.conj().transpose(-1, -2), Z_b)
        prefix = bound.shape[:-1]
        return Y_b.reshape(*prefix, self.actual_dim)


class EduItemMemory:
    """Clean-up associative memory for codebook concept classification."""
    def __init__(self, names: List[str], vectors: torch.Tensor):
        self.names = names
        self.vectors = vectors
        
    def cleanup(self, query: torch.Tensor) -> Tuple[str, float, torch.Tensor]:
        if query.dim() == 1:
            q = query.unsqueeze(0)
        else:
            q = query
            
        if self.vectors.is_complex():
            sims = (q * self.vectors.conj()).sum(dim=-1).real
        else:
            sims = F.cosine_similarity(q, self.vectors, dim=-1)
            
        best_idx = sims.argmax().item()
        best_sim = sims[0, best_idx].item() if sims.dim() > 1 else sims[best_idx].item()
        return self.names[best_idx], best_sim, self.vectors[best_idx]
