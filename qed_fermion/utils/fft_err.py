import torch

def fft2_real_error_analytical(szsz_k_dqmc, szsz_k_dqmc_err):
    """
    Analytical approximation for error propagation.
    """
    # For 2D FFT, the transformation preserves the total "energy"
    # The error scales approximately by sqrt(N) where N = Lx * Lx
    N = szsz_k_dqmc.shape[-1] * szsz_k_dqmc.shape[-2]
    
    # Since we're taking the real part, and assuming independent errors
    # in real and imaginary parts, the error is approximately:
    szsz_r_err = szsz_k_dqmc_err * torch.sqrt(torch.tensor(2.0))  # Factor of sqrt(2) for real+imag
    
    return szsz_r_err