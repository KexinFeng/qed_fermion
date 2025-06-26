import torch
import numpy as np

def reorder_fft_grid2(tensor2d, dims=(-2, -1)):
   """Reorder the last two axes of a tensor from FFT-style to ascending momentum order."""
   Ny, Nx = tensor2d.shape[dims[0]], tensor2d.shape[dims[1]]
   return torch.roll(tensor2d, shifts=(Ny // 2, Nx // 2), dims=dims)

Lx, Ly = 6, 6
vs = Lx * Ly
Ltau = 10

spsm_r = torch.tensor([ 
   7.5751588687437726E-002,
  -5.1221143245789781E-002,
   1.0827749302745038E-003,
  -4.6341290195971475E-002,
  0.12259798197745227     ,
  -10.578708845309418     ,
  -4.0074262318305617E-002,
   7.5426437524042601E-004,
  -9.4053893280412278E-004,
   7.4333650588169280E-004,
  -3.8741540067758908E-002,
   5.1694211695925571E-002,
   9.7525411915148550E-004,
  -7.1846913456576555E-004,
   2.6212542483416850E-005,
  -7.1846913456576555E-004,
   9.7525411915148550E-004,
  -5.1954376961401198E-002,
  -3.8741540067758908E-002,
   7.4333650588169280E-004,
  -9.4053893280412278E-004,
   7.5426437524042590E-004,
  -4.0074262318305617E-002,
   5.1694211695925585E-002,
  0.12259798197745228     ,
  -4.6341290195971482E-002,
   1.0827749302745038E-003,
  -5.1221143245789774E-002,
   7.5751588687437726E-002,
  -10.578708845309418     ,
  -10.548775487328655     ,
   5.4515710785245558E-002,
  -5.6176057827891701E-002,
   5.4515710785245558E-002,
  -10.548775487328655     ,
   90.199212380576370
]).view(Ly, Lx)

spsm_k = torch.fft.ifft2(spsm_r, (Ly, Lx), norm="forward")  # [Ly, Lx]
spsm_k = reorder_fft_grid2(spsm_k) / vs / Ltau  # [Ly, Lx]

dbstop = 1

# Load the binary file as float64 and reshape to (-1, 3)
spsm_bin = np.loadtxt(
   "/Users/kx/Desktop/forked/dqmc_u1sl_mag/run_benchmark/run_meas_J_3_L_6_Ltau_10_bid0_part_0_psz_500_start_5999_end_6000/spsm.bin",
   dtype=np.float64
)

# Extract the third column (index 2)
spsm_bin_col2 = torch.from_numpy(spsm_bin[:, 2])

# Flatten spsm_k for comparison
spsm_k_flat = spsm_k.flatten().real

# Check closeness
torch.testing.assert_close(spsm_bin_col2, spsm_k_flat, rtol=1e-5, atol=1e-5)
print("spsm.bin column 3 matches spsm_k.")


def fourier_trans_eqt(gr, list_, listk, a1_p, a2_p, b1_p, b2_p, filek):
    """
    Python translation of the Fortran subroutine fourier_trans_eqt.
    Args:
        gr: complex tensor, shape [lq]
        list_: integer tensor, shape [lq, 2]
        listk: integer tensor, shape [lq, 2]
        a1_p, a2_p: real vectors, shape [2]
        b1_p, b2_p: real vectors, shape [2]
        filek: output filename
    """
    lq = gr.shape[0]
    gk = torch.zeros(lq, dtype=torch.cdouble)
    # Precompute phase factors
    for imj in range(lq):
        # aimj_p = list(imj,1)*a1_p + list(imj,2)*a2_p
        aimj_p = list_[imj, 0] * a1_p + list_[imj, 1] * a2_p
        for nk in range(lq):
            # xk_p = listk(nk,1)*b1_p + listk(nk,2)*b2_p
            xk_p = listk[nk, 0] * b1_p + listk[nk, 1] * b2_p
            # zexpiqr(imj,nk) = exp(1j * dot(xk_p, aimj_p))
            phase = torch.exp(1j * torch.dot(xk_p, aimj_p))
            gk[nk] += gr[imj] / phase
    gk = gk / lq

    return gk

    # Write to file
    with open(filek, "a") as f:
        for nk in range(lq):
            xk_p = listk[nk, 0] * b1_p + listk[nk, 1] * b2_p
            # Write as 4e16.8: xk_p[0], xk_p[1], gk[nk].real, gk[nk].imag
            f.write(f"{xk_p[0]:16.8e} {xk_p[1]:16.8e} {gk[nk].real:16.8e} {gk[nk].imag:16.8e}\n")



