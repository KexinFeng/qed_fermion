import torch
import numpy as np

def reorder_fft_grid2(tensor2d, dims=(-2, -1)):
   """Reorder the last two axes of a tensor from FFT-style to ascending momentum order."""
   Ny, Nx = tensor2d.shape[dims[0]], tensor2d.shape[dims[1]]
   return torch.roll(tensor2d, shifts=(Ny // 2, Nx // 2), dims=dims)

Lx, Ly = 6, 6
vs = Lx * Ly
Ltau = 10

spsm_r_se = torch.tensor([2.5111786e-01, -2.9179718e-02, -4.0418954e-04, -1.0177700e-04,
        4.5716908e-04, -2.8477559e-02, -2.9097188e-02,  4.5887064e-04,
        4.0912809e-04,  5.8519357e-04, -5.1524254e-05, -1.6313390e-04,
        3.1936483e-04, -8.4689767e-05, -3.8653976e-04,  1.7881603e-04,
       -5.2283780e-05,  1.1808661e-04, -6.3349941e-04, -8.1725520e-05,
       -1.9718637e-04,  4.9818901e-04, -8.7122608e-04,  1.3517730e-04,
       -2.4679257e-04, -4.2752831e-04,  7.8549003e-04,  7.0176284e-05,
       -5.9487886e-04, -5.3164427e-04, -2.9471984e-02,  2.5633283e-04,
       -1.7589578e-04,  8.9146534e-04, -8.0439099e-04, -2.7944249e-04]).view(Ly, Lx)

spsm_r_dqmc = torch.tensor([ 
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

spsm_r_dqmc /= Lx * Ly * Ltau

sum_spsm_r_dqmc = spsm_r_dqmc.mean()
sum_spsm_r_se = spsm_r_se.mean()

dbstop = 1

spsm_k_dqmc = torch.fft.ifft2(spsm_r_dqmc, (Ly, Lx), norm="forward")  # [Ly, Lx]
spsm_k_dqmc = reorder_fft_grid2(spsm_k_dqmc) # [Ly, Lx]
spsm_k_abs_dqmc = spsm_k_dqmc.view(-1).abs()  # [Ly * Lx]
spsm_k_real_dqmc = spsm_k_dqmc.view(-1).real  # [Ly * Lx]
spsm_k_imag_dqmc = spsm_k_dqmc.view(-1).imag  # [Ly * Lx]
# Print the distance between spsm_k_abs_dqmc and spsm_k_real_dqmc
distance = torch.norm(spsm_k_abs_dqmc - spsm_k_real_dqmc)
print("Distance between spsm_k_abs_dqmc and spsm_k_real_dqmc:", distance.item())

spsm_k_se = torch.fft.ifft2(spsm_r_se, (Ly, Lx), norm="forward")  # [Ly, Lx]
spsm_k_se = reorder_fft_grid2(spsm_k_se) # [Ly, Lx]
spsm_k_abs_se = spsm_k_se.view(-1).abs()  # [Ly * Lx]
spsm_k_real_se = spsm_k_se.view(-1).real  # [Ly * Lx]
spsm_k_imag_se = spsm_k_se.view(-1).imag  # [Ly * Lx]
# Print the distance between spsm_k_abs_se and spsm_k_real_se
distance_se = torch.norm(spsm_k_abs_se - spsm_k_real_se)
print("Distance between spsm_k_abs_se and spsm_k_real_se:", distance_se.item())

ky = torch.fft.fftfreq(Ly)
kx = torch.fft.fftfreq(Lx)
ks = torch.stack(torch.meshgrid(ky, kx, indexing='ij'), dim=-1) # [Ly, Lx, (ky, kx)]
ks_ordered = reorder_fft_grid2(ks, dims=(0, 1)).view(-1)  # [Ly, Lx, 2]

dbstop = 1

# # Load the binary file as float64 and reshape to (-1, 3)
# spsm_bin = np.loadtxt(
#    "/Users/kx/Desktop/forked/dqmc_u1sl_mag/run_benchmark/run_meas_J_3_L_6_Ltau_10_bid0_part_0_psz_500_start_5999_end_6000/spsm.bin",
#    dtype=np.float64
# )

# # Extract the third column (index 2)
# spsm_bin_col2 = torch.from_numpy(spsm_bin[:, 2])

# # Flatten spsm_k for comparison
# spsm_k_flat = spsm_k_dqmc.flatten().real

# # Check closeness
# torch.testing.assert_close(spsm_bin_col2, spsm_k_flat, rtol=1e-5, atol=1e-5)
# print("spsm.bin column 3 matches spsm_k.")


# def fourier_trans_eqt(gr, list_, listk, a1_p, a2_p, b1_p, b2_p, filek):
#     """
#     Python translation of the Fortran subroutine fourier_trans_eqt.
#     Args:
#         gr: complex tensor, shape [lq]
#         list_: integer tensor, shape [lq, 2]
#         listk: integer tensor, shape [lq, 2]
#         a1_p, a2_p: real vectors, shape [2]
#         b1_p, b2_p: real vectors, shape [2]
#         filek: output filename
#     """
#     lq = gr.shape[0]
#     gk = torch.zeros(lq, dtype=torch.cdouble)
#     # Precompute phase factors
#     for imj in range(lq):
#         # aimj_p = list(imj,1)*a1_p + list(imj,2)*a2_p
#         aimj_p = list_[imj, 0] * a1_p + list_[imj, 1] * a2_p
#         for nk in range(lq):
#             # xk_p = listk(nk,1)*b1_p + listk(nk,2)*b2_p
#             xk_p = listk[nk, 0] * b1_p + listk[nk, 1] * b2_p
#             # zexpiqr(imj,nk) = exp(1j * dot(xk_p, aimj_p))
#             phase = torch.exp(1j * torch.dot(xk_p, aimj_p))
#             gk[nk] += gr[imj] / phase
#     gk = gk / lq

#     return gk

#     # Write to file
#     with open(filek, "a") as f:
#         for nk in range(lq):
#             xk_p = listk[nk, 0] * b1_p + listk[nk, 1] * b2_p
#             # Write as 4e16.8: xk_p[0], xk_p[1], gk[nk].real, gk[nk].imag
#             f.write(f"{xk_p[0]:16.8e} {xk_p[1]:16.8e} {gk[nk].real:16.8e} {gk[nk].imag:16.8e}\n")



