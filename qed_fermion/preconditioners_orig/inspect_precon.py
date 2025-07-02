from matplotlib import pyplot as plt
import torch
import numpy as np
from matplotlib import colors
plt.ion()
from matplotlib import rcParams
import os
rcParams['figure.raise_window'] = False

Lx = 8
Ltau = 80
file_path = f"./qed_fermion/preconditioners/precon_ckpt_L_{Lx}_Ltau_{Ltau}_dtau_0.1_t_1.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cdtype = torch.complex64  # or torch.float32, depending on your data

precon_dict = torch.load(file_path, map_location=device)
print(f"Loaded preconditioner from {file_path}")

indices = precon_dict["indices"].to(device)
values = precon_dict["values"].to(device)

precon = torch.sparse_coo_tensor(
    indices,
    values,
    size=precon_dict["size"],
    dtype=cdtype,
    device=device
).coalesce()

# precon_csr = precon.to_sparse_csr()

print("Converted to CSR format.")

# Extract main diagonal and second diagonal directly from the sparse COO tensor
row, col = precon.indices()
vals = precon.values()

vs = Lx * Lx
# idx = torch.arange(0, vs * (Ltau - 1), vs, device=device)
d = 3

for rank in list(range(7)) + [Ltau-1, Ltau-2, Ltau-3, Ltau-4, Ltau-5]:  
    idx = torch.arange(0, vs * (Ltau), vs, device=device)

    # Second diagonal: col = row + vs
    second_diag_mask = (col == row + rank * vs)
    second_diag_indices = row[second_diag_mask]
    second_diag_vals = vals[second_diag_mask]

    # Map from index to value for second diagonal
    second_diag_dict = dict(zip(second_diag_indices.tolist(), second_diag_vals.tolist()))
    second_diag_selected = [second_diag_dict.get((i+d).item(), -100) for i in idx]

    print(f"\nRank({rank if rank < 8 else rank - Ltau}) diagonal of precon:")
    print(np.real(second_diag_selected))
    dbstop = 1


if Lx >= 10:
    exit()

#
precon_dense = precon.to_dense().cpu().numpy()
# Define save directory and file name
script_path = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_path, f"./figures")
os.makedirs(save_dir, exist_ok=True)
file_name = "sparse_pattern.pdf"
file_path = os.path.join(save_dir, file_name)

plt.figure(figsize=(8, 8))
plt.spy(precon_dense, markersize=0.5)
plt.title("Sparsity Pattern of Preconditioner")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.show(block=False)
plt.savefig(file_path, format="pdf", bbox_inches="tight")
print(f"Figure saved at: {file_path}")

# Plot the real part of the preconditioner
plt.figure(figsize=(8, 8))
cmap = plt.get_cmap('seismic')
divnorm = colors.TwoSlopeNorm(vmin=np.min(precon_dense.real), vcenter=0.0, vmax=np.max(precon_dense.real))
im = plt.imshow(precon_dense.real, cmap=cmap, norm=divnorm, aspect='auto')
plt.colorbar(im, label='Real Part Value')
plt.title("Real Part of Preconditioner (Dense)")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.show(block=False)

# Save the figure
file_name = "real_value.pdf"
file_path = os.path.join(save_dir, file_name)
plt.savefig(file_path, format="pdf", bbox_inches="tight")
print(f"Figure saved at: {file_path}")

dbstop = 1

#
vs = Lx * Lx
d=3
# Get and print the main diagonal
idx = torch.arange(0, vs * (Ltau), vs, device=device)
main_diag = np.diag(precon_dense)
print("Main diagonal of precon_dense:")
print(main_diag.real[idx+d])

# Get and print the second diagonal (offset=1, just above the main diagonal)
idx = torch.arange(0, vs * (Ltau - 1), vs, device=device)
second_diag = np.diag(precon_dense, k=1*Lx*Lx)
second_diag_dual = np.diag(precon_dense, k=-1*Lx*Lx)
print("Second diagonal (k=1) of precon_dense:")
print(second_diag.real[idx+d])
np.testing.assert_allclose(second_diag.real[idx+d], second_diag_dual.real[idx+d], rtol=1e-5, atol=1e-5)

# For the dense matrix, extract the (Ltau-1)-th diagonal
rank = Ltau - 2
idx = torch.arange(0, vs * (Ltau - (rank)), vs, device=device)
ltau1_diag_dense = np.diag(precon_dense, k=(rank)*Lx*Lx)
ltau1_diag_dense_dual = np.diag(precon_dense, k=-(rank)*Lx*Lx)
print(f"(Ltau-{Ltau-rank})-th diagonal (k={rank}) of precon_dense:")
print(ltau1_diag_dense.real[idx+d])
print(ltau1_diag_dense_dual.real[idx+d])
np.testing.assert_allclose(ltau1_diag_dense.real[idx+d], ltau1_diag_dense_dual.real[idx+d], rtol=1e-5, atol=1e-5)

assert np.all(precon_dense.imag < 1e-6)


dbstop = 1


