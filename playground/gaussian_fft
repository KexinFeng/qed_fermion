import torch
import numpy as np

import sys
sys.path.insert(0, '/Users/kx/Desktop/hmc/qed_fermion')
from qed_fermion.coupling_mat2 import initialize_coupling_mat

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = 'cpu'
print(f"device: {device}")

def fourier_transform(boson):
    """Compute the discrete Fourier transform of the boson field."""
    return torch.fft.fftn(boson, dim=(-3, -2, -1))

def inverse_fourier_transform(boson_k):
    """Compute the inverse discrete Fourier transform."""
    return torch.fft.ifftn(boson_k, dim=(-3, -2, -1)).real

def sample_gaussian(A_eigenvalues, shape):
    """Sample independent Gaussians in Fourier space with variance given by 1 / lambda_k."""
    stddev = torch.sqrt(1.0 / (A_eigenvalues + 1e-8))  # Avoid division by zero
    noise_real = torch.randn(*shape, dtype=torch.float32)
    noise_imag = torch.randn(*shape, dtype=torch.float32)
    noise_k = noise_real + 1j * noise_imag
    return noise_k * stddev

def estimate_covariance_matrix(Lx, Ly, Ltau, num_samples=100):
    """Estimate the covariance matrix by generating samples and computing empirical covariance."""
    shape = (2, Lx, Ly, Ltau)
    cov_matrix = torch.zeros(shape, dtype=torch.float32)
    
    # Generate A's eigenvalues in Fourier space (assumed given for now)
    kx = torch.fft.fftfreq(Lx)
    ky = torch.fft.fftfreq(Ly)
    ktau = torch.fft.fftfreq(Ltau)
    KX, KY, KTAU = torch.meshgrid(kx, ky, ktau, indexing='ij')
    # A_eigenvalues = torch.exp(-(KX**2 + KY**2 + KTAU**2))  # Example eigenvalues (replace with actual ones)
    A_eigenvalues = initialize_coupling_mat(Lx, Ly, Ltau, J=1).to(device)
    
    for _ in range(num_samples):
        phi_k_sampled = sample_gaussian(A_eigenvalues, shape)
        phi_sampled = inverse_fourier_transform(phi_k_sampled)
        cov_matrix += phi_sampled ** 2
    
    return cov_matrix / num_samples

# Example usage
Lx, Ly, Ltau = 10, 10, 10
cov_matrix_estimate = estimate_covariance_matrix(Lx, Ly, Ltau)
print(cov_matrix_estimate.shape)
