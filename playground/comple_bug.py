import torch

# Create a random matrix on MPS
R = torch.tensor([
        # [ 0.2337+0.6519j],
        # [-1.1557+0.1523j],
        # [ 1.3964+0.4732j],
        # [-0.1521-0.4457j],
        # [-0.0533+0.3980j],
        # [-0.4301+0.1241j],
        # [-0.3075+0.7364j],
        # [ 0.7487+0.6936j],
        # [ 0.2189-0.4674j],
        # [-0.2060+0.9202j],
        # [ 1.1810+0.0088j],
        # [ 0.6832-0.8268j],
        # [-0.9786-1.2405j],
        # [ 0.8453-0.6799j],
        # [-0.5651-0.5542j],
        [ 1j]], device='mps')
R_cpu = R.to("cpu")  # Copy to CPU

# Compute R^H R on both devices
mps_result = R.T.conj() @ R
cpu_result = R_cpu.T.conj() @ R_cpu

print("MPS Result:", mps_result)
print("CPU Result:", cpu_result)
print(torch.einsum('ij,ij->', R.conj(), R))
print('-----------')
print(torch.sum(R.conj().T * R.T))
print(torch.sum(R_cpu.abs()**2))

# torch.testing.assert_close(mps_result.to('cpu'), cpu_result)

