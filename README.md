# HMC simulation of U(1) DSL with low-latency CUDA kernels

Paper: [Scalable hybrid quantum Monte Carlo simulation of U(1) gauge field coupled to fermions on GPU](https://scipost.org/SciPostPhys.20.2.060)

We develop a GPU-accelerated hybrid quantum Monte Carlo (QMC) algorithm to solve the fundamental yet
difficult problem of 𝑈(1) gauge field coupled to fermions, which gives rise to a 𝑈(1) Dirac spin liquid state
under the description of (2+1)d quantum electrodynamics QED3. The algorithm renders a good acceptance
rate and, more importantly, nearly linear space-time volume scaling in computational complexity 𝑂(𝑁𝜏𝑉𝑠),
where 𝑁𝜏 is the imaginary time dimension and 𝑉𝑠 is spatial volume, which is much more efficient than
determinant QMC with scaling behavior of 𝑂(𝑁𝜏𝑉𝑠^3). Such acceleration is achieved via a collection of technical
improvements, including (i) the design of the efficient problem-specific preconditioner, (ii) customized CUDA
kernel for matrix-vector multiplication, and (iii) CUDA Graph implementation on the GPU. These advances
allow us to simulate the 𝑈(1) Dirac spin liquid state with unprecedentedly large system sizes, which is up to
𝑁𝜏 × 𝐿 × 𝐿 = 660 × 66 × 66, and reveal its novel properties. With these technical improvements, we see
the asymptotic convergence in the scaling dimensions of various fermion bilinear operators and the conserved
current operator when approaching the thermodynamic limit. The scaling dimensions find good agreement with
field-theoretical expectation, which provides supporting evidence for the conformal nature of the 𝑈(1) Dirac
spin liquid state in the QED3. Our technical advancements open an avenue to study the Dirac spin liquid state
and its transition towards symmetry-breaking phases at larger system sizes and with less computational burden.


Latency vs linear size L:

<img width="506" height="372" alt="Screenshot 2025-08-29 at 12 59 49 PM" src="https://github.com/user-attachments/assets/039a7a7f-1d37-4603-bd92-1cbf8cbbfd10" />

## Dependency repo

The cuda kernels are in repo: 
https://github.com/KexinFeng/cuda_pcg

They need to be compiled with `setup.py` therein and moved to the root directory here.

## How to start

The working execution scripts are in `qed_fermion/pbs_files/`. Typical scripts are:
1. `r_large_cmp.sh`: runs the computation for compact QED model; calls `s_hmc_cmp.cmd` under the hood; write data to `qed_fermion/check_points/hmc_check_point_{suffix}/`
 ```
 bash qed_fermion/pbs_files/r_large_cmp.sh
 ```
2. `r_large_noncmp.sh`: parallel to above except for noncompact QED model.
3. `r_build_cuda.sh`: runs the cuda kernel building code; calls `s_build_cuda.cmd` under the hood, which in turn calls `cr.sh` in repo https://github.com/KexinFeng/cuda_pcg, which in turns calls `setup.py`.
```
bash qed_fermion/pbs_files/r_build_cuda.sh
```

## Environment Setup

### Cuda compilation setup

Specified in `qed_fermion/pbs_files/r_build_cuda.sh`.

### Run time setup

Specified in `qed_fermion/pbs_files/r_large_cmp.sh`, or `qed_fermion/qed_fermion/pbs_files/r_large_noncmp.sh`.


## Postprocessing

The output data is written to `qed_fermion/check_points/hmc_check_point_{suffix}/`.

The data postprocessing scripts and plotting scripts are in `qed_fermion/observables/`.

An entry point of plotting is `qed_fermion/note/merged_panels_corr_cmp.py`.



