import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

class RieFNO(nn.Module):
    """
    hidden_size: dimension of the input channel
    num_blocks: number of block diagonal matrices (higher => fewer parameters but less complexity)
    sparsity_threshold: lambda value for soft shrinkage
    hard_thresholding_fraction: fraction of frequencies to completely mask (lower => reduced computational load)
    """
    def __init__(self, hidden_size, num_blocks=4, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisible by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        # Learnable weight matrices for real and imaginary part transformations
        self.w_real_imag = nn.Parameter(self.scale * torch.randn(3, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b_real_imag = nn.Parameter(self.scale * torch.randn(3, self.num_blocks, self.block_size * self.hidden_size_factor))
        
        # Learnable weight matrices for fusion of real and imaginary features
        self.w_fusion = nn.Parameter(self.scale * torch.randn(3, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b_fusion = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        residual = x  # Save input for residual connection
        # Transpose input to match expected dimensions
        x = torch.transpose(x, 1, 2)

        dtype = x.dtype
        x = x.float()  # Ensure float precision
        B, N, C = x.shape

        # Perform Fourier transform along the temporal dimension
        x = torch.fft.rfft(x, dim=1, norm="ortho")
        x = x.reshape(B, N // 2 + 1, self.num_blocks, self.block_size)

        # Initialize containers for the transformed real and imaginary parts
        o1_real = torch.zeros([B, N // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, N // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        # Intermediate hidden layer
        hidden = torch.zeros([B, N // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)

        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        # Apply transformations on real and imaginary components
        o1_real[:, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].real, self.w_real_imag[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].imag, self.w_real_imag[1]) + \
            self.b_real_imag[0]
        )

        o1_imag[:, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].imag, self.w_real_imag[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].real, self.w_real_imag[1]) + \
            self.b_real_imag[1]
        )

        hidden[:, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].imag + x[:, :kept_modes].real, self.w_real_imag[2]) + \
            self.b_real_imag[2]
        )

        # Combine the real and imaginary parts and hidden features to form output
        o2_real[:, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes], self.w_fusion[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes], self.w_fusion[1]) + \
            torch.einsum('...bi,bio->...bo', hidden[:, :kept_modes], self.w_fusion[2]) + \
            self.b_fusion[0]
        )

        o2_imag[:, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes], self.w_fusion[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes], self.w_fusion[1]) + \
            torch.einsum('...bi,bio->...bo', hidden[:, :kept_modes], self.w_fusion[2]) + \
            self.b_fusion[1]
        )

        # Stack real and imaginary components, apply soft shrinkage for sparsification
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, N // 2 + 1, C)

        # Perform inverse Fourier transform to return to time domain
        x = torch.fft.irfft(x, n=N, dim=1, norm="ortho")
        x = x.type(dtype)

        # Residual connection to maintain input information
        x = torch.transpose(x, 1, 2)
        return x + residual
