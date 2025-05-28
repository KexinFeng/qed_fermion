import torch
import sys
import os

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../')

import numpy as np

from qed_fermion.hmc_sampler_batch import HmcSampler
from qed_fermion.stochastic_estimator import StochaticEstimator


def test_green_functions():
    hmc = HmcSampler()
    hmc.Lx = 4
    hmc.Ly = 4
    hmc.Ltau = 2

    hmc.bs = 2
    hmc.reset()
    hmc.initialize_boson_pi_flux_randn_matfree()
    boson = hmc.boson[1].unsqueeze(0)
    hmc.bs = 1

    se = StochaticEstimator(hmc)
    se.Nrv = 100_000  # bs > 10000 will fail on _C.mh_vec, due to grid = {Ltau, bs}.
    se.Nrv = 200  # bs >= 80 will fail on cuda _C.prec_vec. This is size independent
    se.Nrv = 20  # minum Nrv to pass assertions
    
    se.test_orthogonality(se.random_vec_bin())
    # se.test_orthogonality(se.random_vec_norm())
    se.test_fft_negate_k3()

    # Compute Green prepare
    eta = se.random_vec_bin()  # [Nrv, Ltau * Ly * Lx]
    # eta = se.random_vec_norm().to(torch.complex64)  # [Nrv, Ltau * Ly * Lx]

    se.set_eta_G_eta_debug(boson, eta)
    Gij_gt = se.G_groundtruth(boson)

    # Test Green
    G_stoch = se.G_delta_0()
    G_stoch_primitive = se.G_delta_0_primitive()
    torch.testing.assert_close(G_stoch.real, G_stoch_primitive.real, rtol=1e-2, atol=5e-2)
    
    G_gt = se.G_delta_0_groundtruth(Gij_gt)
    torch.testing.assert_close(G_gt.real, G_stoch_primitive.real, rtol=1e-2, atol=5e-2)

    # Test Green extended
    G_stoch_ext = se.G_delta_0_ext()
    G_stoch_primitive_ext = se.G_delta_0_primitive_ext()
    torch.testing.assert_close(G_stoch_ext.real, G_stoch_primitive_ext.real, rtol=1e-2, atol=5e-2)

    G_gt_ext = se.G_delta_0_groundtruth_ext(Gij_gt)
    torch.testing.assert_close(G_gt_ext.real, G_stoch_primitive_ext.real, rtol=1e-2, atol=5e-2)

    # Test Green four-point
    GG_stoch_primitive = se.G_delta_0_G_delta_0_primitive()
    GG_gt = se.G_delta_0_G_delta_0_groundtruth(Gij_gt)
    GG_stoch = se.G_delta_0_G_delta_0()
    torch.testing.assert_close(GG_stoch_primitive, GG_gt, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(GG_stoch, GG_stoch_primitive, rtol=1e-2, atol=2e-2)

    # Test Green four-point GG_D0D0 extended
    GG_ext = se.G_delta_0_G_delta_0_ext()
    GG_primitive_ext = se.G_delta_0_G_delta_0_primitive_ext()
    GG_gt_ext = se.G_delta_0_G_delta_0_groundtruth_ext(Gij_gt)
    torch.testing.assert_close(GG_primitive_ext, GG_gt_ext, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(GG_primitive_ext, GG_ext, rtol=1e-2, atol=2e-2)

    # Test Green four-point GG_DD00 extended
    GG_ext = se.G_delta_delta_G_0_0_ext()
    GG_gt_ext = se.G_delta_delta_G_0_0_groundtruth_ext(Gij_gt)
    torch.testing.assert_close(GG_gt_ext, GG_ext, rtol=1e-2, atol=2e-2)

    # Test Green four-point GG_DD00 extended
    GG_ext = se.G_delta_0_G_0_delta_ext()
    GG_gt_ext = se.G_delta_0_G_0_delta_groundtruth_ext(Gij_gt)
    torch.testing.assert_close(GG_gt_ext, GG_ext, rtol=1e-2, atol=2e-2)
 

    print("✅ All assertions pass!")


def test_fermion_obsr():
    hmc = HmcSampler()
    hmc.Lx = 6
    hmc.Ly = 6
    hmc.Ltau = 240

    hmc.bs = 2
    hmc.reset()
    hmc.initialize_boson_pi_flux_randn_matfree()

    se = StochaticEstimator(hmc, cuda_graph_se=True)
    se.Nrv = 200  # bs >= 80 will fail on cuda _C.prec_vec. This is size independent
    se.init_cuda_graph()

    # Compute Green prepare
    eta = se.random_vec_bin()  # [Nrv, Ltau * Ly * Lx]

    bosons = hmc.boson
    if se.cuda_graph_se:
        obsr = se.graph_runner(bosons, eta)
    else:
        obsr = se.get_fermion_obsr(bosons, eta)

    obsr_ref = se.get_fermion_obsr(bosons, eta)
    torch.testing.assert_close(obsr['spsm_r'], obsr_ref['spsm_r'], rtol=1e-2, atol=5e-2)
    print()


def test_fermion_obsr_write():
    hmc = HmcSampler()
    hmc.Lx = 6
    hmc.Ly = 6
    hmc.Ltau = 10

    hmc.bs = 5
    hmc.reset()
    hmc.initialize_boson_pi_flux_randn_matfree()

    se = StochaticEstimator(hmc, cuda_graph_se=True)
    se.Nrv = 200  # bs >= 80 will fail on cuda _C.prec_vec. This is size independent
    se.init_cuda_graph()

    # Compute Green prepare
    eta = se.random_vec_bin()  # [Nrv, Ltau * Ly * Lx]

    bosons = hmc.boson
    if se.cuda_graph_se:
        obsr = se.graph_runner(bosons, eta)
    else:
        obsr = se.get_fermion_obsr(bosons, eta)

    obsr_ref = se.get_fermion_obsr(bosons, eta)
    torch.testing.assert_close(obsr['spsm_r'], obsr_ref['spsm_r'], rtol=1e-2, atol=5e-2)
    print()

    # ---------- Benchmark vs dqmc ---------- #
    Lx, Ly, Ltau = hmc.Lx, hmc.Ly, hmc.Ltau
    J = 0.5
    bs = 5
    assert bs == hmc.bs, "Batch size mismatch."
    input_folder = "/Users/kx/Desktop/hmc/fignote/ftdqmc/benchmark_6x6x10_bs5/hmc_check_point_6x10/"
    input_folder = f"/users/4/fengx463/hmc/qed_fermion/qed_fermion/post_processors/fermi_bench/"
    hmc_filename = f"/stream_ckpt_N_hmc_{Lx}_Ltau_{Ltau}_Nstp_6000_bs{bs}_Jtau_{J:.2g}_K_1_dtau_0.1_delta_t_0.05_N_leapfrog_4_m_1_step_6000.pt"

    # Parse to get specifics
    path_parts = hmc_filename.split('/')
    filename = path_parts[-1]
    filename_parts = filename.split('_')
    specifics = '_'.join(filename_parts[1:]).replace('.pt', '')
    print(f"Parsed specifics: {specifics}")

    parts = hmc_filename.split('_')
    jtau_index = parts.index('Jtau')  # Find position of 'Jtau'
    jtau_value = float(parts[jtau_index + 1])   # Get the next element
    
    # Load and write
    # [seq, Ltau * Ly * Lx * 2]
    boson_seq = torch.load(input_folder + hmc_filename)
    # boson_seq = boson_seq.to(device='mps', dtype=torch.float32)
    print(f'Loaded: {input_folder + hmc_filename}')        
    
    # Extract Nstep and Nstep_local from filenames
    # hmc_match = re.search(r'Nstp_(\d+)', hmc_filename)
    # end = int(hmc_match.group(1))
    end = 6000

    start = 5999
    sample_step = 1
    seq_idx = set(list(range(start, end, sample_step)))

    # Write result to file
    filtered_seq = [(i, boson) for i, boson in enumerate(boson_seq) if i in seq_idx]
    spsm_k = torch.zeros((len(filtered_seq), bs, Ly, Lx), dtype=hmc.dtype)
    for i, boson in filtered_seq:
        print(f"boson shape: {boson[1].shape}, dtype: {boson[1].dtype}, device: {boson[1].device}")

        if se.cuda_graph_se:
            obsr = se.graph_runner(bosons, eta)
        else:
            obsr = se.get_fermion_obsr(bosons, eta)
        spsm_k[i-start] = obsr['spsm_k_abs']  # [bs, Ly, Lx]
    
    # ks = obsr['ks']  # [Lx, Ly, 2]
    ks = se.get_ks_ordered()  # [Ly, Lx, (ky, kx)]

    # Linearize
    spsm_k_mean = spsm_k.mean(dim=(0))  # [bs, Ly, Lx]
    spsm_k_mean = spsm_k_mean.reshape(bs, -1)  # Ly*Lx
    ks = ks.reshape(-1, 2).flip(dims=[-1])  # Ly*Lx, but displayed as (kx, ky)
    print("ks (flattened):", ks)

    for b in range(bs):
        print(f"spsm_k_mean (flattened) bid:{b}: ", spsm_k_mean[b])

    output_dir = os.path.join(script_path, "../post_processors/fermi_bench/Nrv_{se.Nrv}")
    os.makedirs(output_dir, exist_ok=True)
    for b in range(bs):
        data = torch.stack([ks[:, 0], ks[:, 1], spsm_k_mean[b]], dim=1).cpu().numpy()
        output_file = os.path.join(output_dir, f"spsm_k_b{b}.txt")
        # Save as text, columns: kx, ky, spsm
        np.savetxt(output_file, data, fmt="%.8f", comments='')
        print(f"Saved: {output_file}")

    print("✅ All assertions pass!")


if __name__ == "__main__":
    # test_green_functions()
    test_fermion_obsr_write()
    # test_fermion_obsr()
    print("All tests completed successfully!")
