import torch
import sys

sys.path.insert(0, '/Users/kx/Desktop/hmc/qed_fermion')
from qed_fermion.coupling_mat3 import initialize_curl_mat
from qed_fermion.hmc_sampler2_batch_fermion import HmcSampler
from unit_test.utils import clear_mat

def action_tau(x, Ltau, coeff):
    s = 0
    for tau in range(Ltau):
        s += (x[..., (tau+1)%Ltau] - x[..., tau])**2
    return coeff * s.sum().view(-1)

def test_action_tau_cmp():
    hmc = HmcSampler()

    Lx, Ly, Ltau = 2, 2, 4
    hmc.Lx, hmc.Ly, hmc.Ltau = Lx, Ly, Ltau
    hmc.reset()

    dphis = torch.tensor([1e-2, 1e-3, 1e-4, 5e-4])
    daction = []
    action = []
    x = torch.randn(1, 2, Lx, Ly, Ltau)
    for dphi in dphis:
        phi_tau = torch.arange(0, Ltau) * dphi
        x[..., :] = phi_tau
    
        action1 = hmc.action_boson_tau_cmp(x)
        action2 = action_tau(x, Ltau, 1/2 / hmc.J / hmc.dtau**2)

        action.append(action1 / dphi**2)
        daction.append(action2 / dphi**2 )

    print(action)
    print(daction)
    

test_action_tau_cmp()