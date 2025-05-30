from matplotlib import pyplot as plt
import torch
import numpy as np
from matplotlib import colors
plt.ion()
from matplotlib import rcParams
rcParams['figure.raise_window'] = False

Lx = 6
Ltau = 120
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

precon_csr = precon.to_sparse_csr()

print("Converted to CSR format.")

#
precon_dense = precon_csr.to_dense().cpu().numpy()
plt.figure(figsize=(8, 8))
plt.spy(precon_dense, markersize=0.5)
plt.title("Sparsity Pattern of Preconditioner")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.show(block=False)

plt.figure(figsize=(8, 8))
# Choose a diverging colormap with white at zero
cmap = plt.get_cmap('seismic')
# Center the colorbar at zero, set white at zero
divnorm = colors.TwoSlopeNorm(vmin=np.min(precon_dense.real), vcenter=0.0, vmax=np.max(precon_dense.real))
im = plt.imshow(precon_dense.real, cmap=cmap, norm=divnorm, aspect='auto')
plt.colorbar(im, label='Real Part Value')
plt.title("Real Part of Preconditioner (Dense)")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.show(block=False)

dbstop = 1

#
vs = Lx * Lx
idx = torch.arange(0, vs * (Ltau - 1), vs, device=device)
d=2
# Get and print the main diagonal
main_diag = np.diag(precon_dense)
print("Main diagonal of precon_dense:")
print(main_diag.real[idx+d])

# Get and print the second diagonal (offset=1, just above the main diagonal)
second_diag = np.diag(precon_dense, k=1*Lx*Lx)
print("Second diagonal (k=1) of precon_dense:")
print(second_diag.real[idx+d])


dbstop = 1


