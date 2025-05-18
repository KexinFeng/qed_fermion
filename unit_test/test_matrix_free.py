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
    
    # Initialize boson field randomly
    hmc.bs = 2  # Batch size
    hmc.initialize_boson()
    boson = hmc.boson
    
    # Test 1: Compare action_boson_plaq vs action_boson_plaq_matfree
    action_matrix = hmc.action_boson_plaq(boson)
    action_matfree = hmc.action_boson_plaq_matfree(boson)
    
    # Check if they produce the same result
    print(f"Action with matrix: {action_matrix}")
    print(f"Action matrix-free: {action_matfree}")
    torch.testing.assert_close(action_matrix, action_matfree, atol=1e-5, rtol=1e-5)
    print("✓ Action test passed!")
    
    # Test 2: Compare force_b_plaq vs force_b_plaq_matfree
    force_matrix = hmc.force_b_plaq(boson)
    force_matfree = hmc.force_b_plaq_matfree(boson)
    
    # Check if they produce the same result
    print(f"Force L2 norm (matrix): {torch.norm(force_matrix)}")
    print(f"Force L2 norm (matfree): {torch.norm(force_matfree)}")
    print(f"Difference L2 norm: {torch.norm(force_matrix - force_matfree)}")
    torch.testing.assert_close(force_matrix, force_matfree, atol=1e-5, rtol=1e-5)
    print("✓ Force test passed!")
    
    # Test with staggered field
    print("\nTesting with staggered pi field:")
    hmc.initialize_boson_staggered_pi()
    boson = hmc.boson
    
    # Compare actions
    action_matrix = hmc.action_boson_plaq(boson)
    action_matfree = hmc.action_boson_plaq_matfree(boson)
    print(f"Action with matrix: {action_matrix}")
    print(f"Action matrix-free: {action_matfree}")
    torch.testing.assert_close(action_matrix, action_matfree, atol=1e-5, rtol=1e-5)
    print("✓ Staggered action test passed!")
    
    # Compare forces
    force_matrix = hmc.force_b_plaq(boson)
    force_matfree = hmc.force_b_plaq_matfree(boson)
    print(f"Force L2 norm (matrix): {torch.norm(force_matrix)}")
    print(f"Force L2 norm (matfree): {torch.norm(force_matfree)}")
    print(f"Difference L2 norm: {torch.norm(force_matrix - force_matfree)}")
    torch.testing.assert_close(force_matrix, force_matfree, atol=1e-5, rtol=1e-5)
    print("✓ Staggered force test passed!")
    
if __name__ == "__main__":
    test_matrix_free_implementations()
