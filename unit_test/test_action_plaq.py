import torch
import sys

sys.path.insert(0, '/Users/kx/Desktop/hmc/qed_fermion')
from qed_fermion.coupling_mat3 import initialize_curl_mat
from qed_fermion.hmc_sampler2_batch_fermion import HmcSampler
from unit_test.utils import clear_mat


def test_action_plaq():
    hmc = HmcSampler()

    Lx, Ly, Ltau = 2, 2, 2
    hmc.Lx, hmc.Ly, hmc.Ltau = Lx, Ly, Ltau
    hmc.curl_mat = initialize_curl_mat(Lx, Ly).to(hmc.curl_mat.device)
    hmc.initialize_geometry()

    # staggered boson
    hmc.initialize_boson_staggered_pi()
    boson = hmc.boson
    assert (hmc.action_boson_plaq(boson) == -4).all()

    hmc.initialize_boson()
    boson = hmc.boson
    energy = hmc.action_boson_plaq(boson)



test_action_plaq()
