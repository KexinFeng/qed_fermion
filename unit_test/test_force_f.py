import torch
import sys

sys.path.insert(0, '/Users/kx/Desktop/hmc/qed_fermion')
from qed_fermion.utils.coupling_mat3 import initialize_curl_mat
from qed_fermion.hmc_sampler_batch import HmcSampler
from unit_test.utils import clear_mat


# @torch.inference_mode()
def test_force_f():
    hmc = HmcSampler()

    Lx, Ly, Ltau = 2, 2, 2
    Lx, Ly, Ltau = 6, 6, 10
    hmc.Lx, hmc.Ly, hmc.Ltau = Lx, Ly, Ltau
    hmc.curl_mat = initialize_curl_mat(Lx, Ly).to(hmc.curl_mat.device)
    hmc.initialize_geometry()

    # staggered boson
    hmc.initialize_boson_staggered_pi()
    boson = hmc.boson
    M, B_list = hmc.get_M(boson)

    R = hmc.draw_psudo_fermion()
    psi = torch.einsum('rs,bs->br', M.T.conj(), R)

    force_fs, _ = hmc.force_f([psi], M, boson, B_list)
    force_fs[0].unsqueeze_(0)

    with torch.inference_mode(False):
        boson = boson.clone().requires_grad_(True)
        M_auto, _ = hmc.get_M(boson)
        Ot = M_auto.conj().T @ M_auto
        L = torch.linalg.cholesky(Ot)
        O_inv = torch.cholesky_inverse(L) 
        Sf = torch.einsum('bi,ij,bj->b', psi.conj(), O_inv, psi)
        torch.testing.assert_close(torch.imag(Sf), torch.zeros_like(torch.imag(Sf)))
        Sf = torch.real(Sf)
        force_f_auto = -torch.autograd.grad(Sf, boson, create_graph=False)[0]
    
    print(force_fs[0].permute([0, 4, 3, 2, 1]).view(-1))
    print(force_f_auto.permute([0, 4, 3, 2, 1]).view(-1))

    torch.testing.assert_close(clear_mat(force_fs[0]), clear_mat(force_f_auto))

    # random boson
    hmc.initialize_boson()
    boson = hmc.boson
    M, B_list = hmc.get_M(boson)

    R = hmc.draw_psudo_fermion()
    psi = torch.einsum('rs,bs->br', M.T.conj(), R)

    force_fs, _ = hmc.force_f([psi], M, boson, B_list)
    force_fs[0].unsqueeze_(0)

    with torch.inference_mode(False):
        boson = boson.clone().requires_grad_(True)
        M_auto, _ = hmc.get_M(boson)
        Ot = M_auto.conj().T @ M_auto
        L = torch.linalg.cholesky(Ot)
        O_inv = torch.cholesky_inverse(L) 
        Sf = torch.einsum('bi,ij,bj->b', psi.conj(), O_inv, psi)
        torch.testing.assert_close(torch.imag(Sf), torch.zeros_like(torch.imag(Sf)))
        Sf = torch.real(Sf)
        force_f_auto = -torch.autograd.grad(Sf, boson, create_graph=False)[0]
    
    print('--')
    print(force_fs[0].permute([0, 4, 3, 2, 1]).view(-1))
    print(force_f_auto.permute([0, 4, 3, 2, 1]).view(-1))

    torch.testing.assert_close(clear_mat(force_fs[0]), clear_mat(force_f_auto), atol=1e-4, rtol=1e-5)

            

if __name__ == '__main__':
    test_force_f()
