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
    Lx, Ly, Ltau = 6, 6, 10
    hmc.Lx, hmc.Ly, hmc.Ltau = Lx, Ly, Ltau
    hmc.curl_mat = initialize_curl_mat(Lx, Ly).to(hmc.curl_mat.device)
    hmc.initialize_geometry()

    # random boson
    hmc.initialize_boson()
    boson = hmc.boson
    M, B_list = hmc.get_M(boson)

    R = hmc.draw_psudo_fermion()
    psi = torch.einsum('rs,bs->br', M.T.conj(), R)

    force_fs, _ = hmc.force_f(psi, M, boson, B_list)
    force_f = force_fs[0].unsqueeze_(0)

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
    print(force_f.permute([0, 4, 3, 2, 1]).view(-1))
    print(force_f_auto.permute([0, 4, 3, 2, 1]).view(-1))

    torch.testing.assert_close(clear_mat(force_f), clear_mat(force_f_auto), atol=1e-4, rtol=1e-5)

    # Manual diff
    for xy in range(2):
        for i in range(Lx):
            for j in range(Ly):
                for t in range(Ltau):
                    boson2 = boson.clone()
                    dx = 0.001
                    boson2[0, xy, i, j, t] = boson2[0, xy, i, j, t] + dx
                    M2, _ = hmc.get_M(boson2)
                    # psi2 = torch.einsum('rs,bs->br', M2.T.conj(), R)
                    
                    Ot = M2.conj().T @ M2
                    L = torch.linalg.cholesky(Ot)
                    O_inv = torch.cholesky_inverse(L) 
                    Sf2 = torch.real(torch.einsum('bi,ij,bj->b', psi.conj(), O_inv, psi)) 

                    # derivative
                    force_f_0 = -(Sf2 - Sf)/dx

                    # force_f = force_f.permute([0, 4, 3, 2, 1]).view(-1)
                    # force_f_auto = force_f_auto.permute([0, 4, 3, 2, 1]).view(-1)

                    print(f'force_f_man: {force_f_0}, \nforce_cal={force_f[0, xy, i, j, t]}\n')
                    # print(f'force_auto={force_f_auto[0, xy, i, j, t]}')

                    dbstop = 0
            

if __name__ == '__main__':
    test_force_f()

