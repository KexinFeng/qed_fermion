import torch
import sys

sys.path.insert(0, '/Users/kx/Desktop/hmc/qed_fermion')
from qed_fermion.hmc_sampler_batch import HmcSampler

torch.manual_seed(0)

@torch.inference_mode()
def test():
    # torch.dot((x - x_last).view(-1), (force_b_plaq + force_b_tau).reshape(-1))
    # tensor(-0.0107)
    # self.action_boson_plaq(x) + self.action_boson_tau_cmp(x) - self.action_boson_plaq(x_last) - self.action_boson_tau_cmp(x_last)
    # tensor([0.0053])

    # Load the tensors from the file
    checkpoint = torch.load('tmp.pt')
    x = checkpoint['x']
    x_last = checkpoint['x_last']

    # Initialize the HMC class
    hmc = HmcSampler()

    # Initialize x and x_last with close values
    x = torch.randn_like(x)
    x_last = x.clone()

    # Loop through all elements of x
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                for l in range(x.shape[3]):
                    for m in range(x.shape[4]):
                        x_last[i, j, k, l, m] += 0.01

                        # Compute force_b_tau_manual
                        force_b_tau_manual = hmc.force_b_tau_cmp(x)

                        # Check the dot product
                        dot_product = -torch.dot((x_last - x).view(-1), (force_b_tau_manual).reshape(-1))[None]
                        action_difference = hmc.action_boson_tau_cmp(x_last) - hmc.action_boson_tau_cmp(x)

                        print(dot_product, action_difference)
                        # Use torch.testing.assert_close for comparison
                        torch.testing.assert_close(dot_product, action_difference, atol=1e-2, rtol=1e-3)

                        # Reset x_last for the next iteration
                        x_last[i, j, k, l, m] -= 0.01


def test_auto_d1():
    # torch.dot((x - x_last).view(-1), (force_b_plaq + force_b_tau).reshape(-1))
    # tensor(-0.0107)
    # self.action_boson_plaq(x) + self.action_boson_tau_cmp(x) - self.action_boson_plaq(x_last) - self.action_boson_tau_cmp(x_last)
    # tensor([0.0053])

    # Load the tensors from the file
    checkpoint = torch.load('tmp.pt')
    x = checkpoint['x']
    x_last = checkpoint['x_last']

    # Initialize the HMC class
    hmc = HmcSampler()

    # Initialize x and x_last with close values
    x = torch.randn_like(x)
    x_last = x.clone()

    # Loop through all elements of x
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                for l in range(x.shape[3]):
                    for m in range(x.shape[4]):
                        x_last[i, j, k, l, m] += 0.01

                        # Compute force_b_tau_manual
                        # force_b_tau_manual = hmc.force_b_tau_cmp(x)

                        with torch.enable_grad():
                            assert x.grad is None
                            x_grad = x.clone().requires_grad_(True)
                            Sb_tau = hmc.action_boson_tau_cmp(x_grad)
                            force_b_tau_manual = -torch.autograd.grad(Sb_tau, x_grad, create_graph=False)[0]

                        # Check the dot product
                        dot_product = -torch.dot((x_last - x).view(-1), (force_b_tau_manual).reshape(-1))[None]
                        action_difference = hmc.action_boson_tau_cmp(x_last) - hmc.action_boson_tau_cmp(x)

                        print(dot_product, action_difference)
                        # Use torch.testing.assert_close for comparison
                        torch.testing.assert_close(dot_product, action_difference, atol=1e-2, rtol=1e-3)

                        # Reset x_last for the next iteration
                        x_last[i, j, k, l, m] -= 0.01

def test_auto_d_vs_manual_d():
    # Load the tensors from the file
    checkpoint = torch.load('tmp.pt')
    x = checkpoint['x']


    # Initialize the HMC class
    hmc = HmcSampler()

    # Compute forces
    with torch.enable_grad():
        assert x.grad is None
        Sb_tau = hmc.action_boson_tau_cmp(x)
        force_b_tau = -torch.autograd.grad(Sb_tau, x, create_graph=False)[0]
    
    force_b_tau_manual = hmc.force_b_tau_cmp(x)

    torch.testing.assert_close(force_b_tau, force_b_tau_manual, atol=1e-2, rtol=1e-3)


test()
test_auto_d1()
test_auto_d_vs_manual_d()


