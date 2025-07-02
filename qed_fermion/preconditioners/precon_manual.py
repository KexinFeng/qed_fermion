import torch

Lx = 10
Ltau = 100
Ly = Lx
vs = Lx * Ly
file_path = f"./qed_fermion/preconditioners/man_precon_ckpt_L_{Lx}_Ltau_{Ltau}_dtau_0.1_t_1.pt"
size = vs * Ltau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cdtype = torch.complex64  

# indices # [2, nnz]
# values # [2, nnz]

out_values = []
out_indices = []

# Rank 0
# Edge values
edge_values = [
    0.91698223, 1.02107012, 1.1740706,  1.24646831, 1.28092396, 1.29741096,
    1.30534041, 1.30917168, 1.3110323,  1.31193936
]

# Tail values
tail_values = [
    1.31167626, 1.31074941, 1.30885756, 1.30498219, 1.29699361, 1.28042281,
    1.24588311, 1.17343438, 1.02052784
]

# Number of edge and tail values
n_edge = len(edge_values)
n_tail = len(tail_values)

# Number of middle values to fill
n_middle = Ltau - n_edge - n_tail

# Repeat the middle value
middle_value = 1.312381281
middle_values = torch.full((n_middle,), middle_value, dtype=cdtype, device=device)

values = torch.cat([
    torch.tensor(edge_values, dtype=cdtype, device=device),
    middle_values,
    torch.tensor(tail_values, dtype=cdtype, device=device)
])
# Repeat each value in 'values' vs times and interleave them
values = values.repeat_interleave(vs)
indices = torch.arange(size, device=device)
indices = torch.stack([indices, indices], dim=0)

out_values.append(values)
out_indices.append(indices)


# Test
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

precon_man = torch.sparse_coo_tensor(
    torch.cat(out_indices, dim=1),
    torch.cat(out_values, dim=0),
    size=(size, size),
    dtype=cdtype,
    device=device
).coalesce()
print("precon_man indices:", precon_man.indices())
print("precon_man values:", precon_man.values())
print("precon_man size:", precon_man.size())

# Rank 1






dbstop = 1


