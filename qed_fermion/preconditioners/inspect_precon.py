import torch

file_path = "/home/fengx463/hmc/qed_fermion/qed_fermion/preconditioners/precon_ckpt_L_6_Ltau_60_dtau_0.1_t_1.pt"
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




