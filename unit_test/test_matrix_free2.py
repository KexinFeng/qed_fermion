import torch
import sys

sys.path.insert(0, '/Users/kx/Desktop/hmc/qed_fermion')
from qed_fermion.hmc_sampler_batch import HmcSampler
from qed_fermion.utils.coupling_mat3 import initialize_curl_mat

def test_matrix_free_implementations():
    # Initialize HMC sampler
    hmc = HmcSampler()
    
    # Set dimensions
    Lx, Ly, Ltau = 6, 6, 10
    hmc.Lx, hmc.Ly, hmc.Ltau = Lx, Ly, Ltau
    hmc.initialize_curl_mat()  # This should set up curl_mat automatically
    hmc.initialize_geometry()  # Set up geometry
    hmc.K = 1.0  # Set K value
    
    
    # ------------------------------------------------------------
    hmc.bs = 1  # Batch size
    hmc.reset()
    # Test boson initialization equivalence: random + staggered
    print("\nTesting boson initialization equivalence...")
    # hmc.test_initialize_boson_equivalence()
    hmc.initialize_boson()
    boson_old = hmc.boson.clone()
    hmc.initialize_boson_matfree()
    boson_new = hmc.boson_matfree
    torch.testing.assert_close(boson_old, boson_new, atol=1e-5, rtol=1e-5)
    print("✓ test_initialize_boson_equivalence passed!")
    # hmc.test_initialize_boson_staggered_pi_equivalence()
    hmc.initialize_boson_staggered_pi()
    boson_old = hmc.boson.clone()
    hmc.initialize_boson_staggered_pi_matfree()
    boson_new = hmc.boson_matfree
    torch.testing.assert_close(boson_old, boson_new, atol=1e-5, rtol=1e-5)
    print("✓ test_initialize_boson_staggered_pi_equivalence passed!")
    
if __name__ == "__main__":
    test_matrix_free_implementations()
