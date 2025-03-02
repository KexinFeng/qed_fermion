import torch
import sys

sys.path.insert(0, '/Users/kx/Desktop/hmc/qed_fermion')
from qed_fermion.coupling_mat3 import initialize_curl_mat
from qed_fermion.hmc_sampler2_batch_fermion import HmcSampler
from unit_test.utils import clear_mat

def action_boson_plaq(boson):
    """
    boson: [bs, 2, Lx, Ly, Ltau]
    curl phi = phix(r) + phiy(r+x) - phix(r+y) - phiy(r)
    """
    s = 0
    bs, _, Lx, Ly, Ltau = boson.shape
    for x in range(Lx):
        for y in range(Ly):
            for tau in range(Ltau):
                s += torch.cos(
                    boson[0, 0, x, y, tau] + boson[0, 1, (x+1)%Lx, y, tau] \
                - boson[0, 0, x, (y+1)%Ly, tau] - boson[0, 1, x, y, tau]
                )
    return s

def test_action_plaq():
    hmc = HmcSampler()

    Lx, Ly, Ltau = 2, 2, 2
    hmc.Lx, hmc.Ly, hmc.Ltau = Lx, Ly, Ltau
    hmc.curl_mat = initialize_curl_mat(Lx, Ly).to(hmc.curl_mat.device)
    hmc.initialize_geometry()

    # staggered boson
    hmc.initialize_boson_staggered_pi()
    boson = hmc.boson
    assert (hmc.action_boson_plaq(boson) == -8 * hmc.K).all()

    hmc.initialize_boson()
    boson = hmc.boson
    assert (hmc.action_boson_plaq(boson) == hmc.K * action_boson_plaq(boson)).all()



test_action_plaq()
