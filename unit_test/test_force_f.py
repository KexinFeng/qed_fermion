import torch
import sys

sys.path.insert(0, '/Users/kx/Desktop/hmc/qed_fermion')
from qed_fermion.coupling_mat3 import initialize_curl_mat
from qed_fermion.hmc_sampler2_batch_fermion import HmcSampler
from unit_test.utils import clear_mat


# @torch.inference_mode()
def test_force_f():
    hmc = HmcSampler()

    Lx, Ly, Ltau = 2, 2, 2
    hmc.Lx, hmc.Ly, hmc.Ltau = Lx, Ly, Ltau
    hmc.curl_mat = initialize_curl_mat(Lx, Ly).to(hmc.curl_mat.device)
    hmc.initialize_geometry()

    # staggered boson
    hmc.initialize_boson_staggered_pi()
    boson = hmc.boson
    M = hmc.get_M(boson)

    R = hmc.draw_psudo_fermion()
    psi = torch.einsum('rs,bs->br', M.T.conj(), R)

    force_f, _ = hmc.force_f(psi, M, boson)
    force_f.unsqueeze_(0)

    with torch.inference_mode(False):
        boson = boson.clone().requires_grad_(True)
        M_auto = hmc.get_M(boson)
        Ot = M_auto.conj().T @ M_auto
        L = torch.linalg.cholesky(Ot)
        O_inv = torch.cholesky_inverse(L) 
        Sf = torch.einsum('bi,ij,bj->b', psi.conj(), O_inv, psi)
        torch.testing.assert_close(torch.imag(Sf), torch.zeros_like(torch.imag(Sf)))
        Sf = torch.real(Sf)
        force_f_auto = -torch.autograd.grad(Sf, boson, create_graph=False)[0]
    
    print(force_f.permute([0, 4, 3, 2, 1]).view(-1))
    print(force_f_auto.permute([0, 4, 3, 2, 1]).view(-1))

    torch.testing.assert_close(clear_mat(force_f), clear_mat(force_f_auto))


    # random boson
    hmc.initialize_boson()
    boson = hmc.boson
    M = hmc.get_M(boson)

    R = hmc.draw_psudo_fermion()
    psi = torch.einsum('rs,bs->br', M.T.conj(), R)

    force_f = hmc.force_f(psi, M, boson)

    with torch.inference_mode(False):
        boson = boson.clone().requires_grad_(True)
        M = hmc.get_M(boson)
        Sf = torch.einsum('bi,ij,bj->b', psi.conj(), M, psi)
        force_f_auto = torch.autograd.grad(Sf, boson, create_graph=False)[0]
    
    torch.testing.assert_close(force_f, force_f_auto)  
            

if __name__ == '__main__':
    test_force_f()
