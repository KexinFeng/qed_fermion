# Scalable Hybrid quantum Monte Carlo simulation of U(1) gauge field coupled to fermions on GPU

https://arxiv.org/pdf/2508.16298

We develop a GPU-accelerated hybrid quantum Monte Carlo (QMC) algorithm to solve the fundamental yet
difficult problem of 𝑈(1) gauge field coupled to fermions, which gives rise to a 𝑈(1) Dirac spin liquid state
under the description of (2+1)d quantum electrodynamics QED3. The algorithm renders a good acceptance
rate and, more importantly, nearly linear space-time volume scaling in computational complexity 𝑂(𝑁𝜏𝑉𝑠),
where 𝑁𝜏 is the imaginary time dimension and 𝑉𝑠 is spatial volume, which is much more efficient than
determinant QMC with scaling behavior of 𝑂(𝑁𝜏𝑉
3
𝑠
). Such acceleration is achieved via a collection of technical
improvements, including (i) the design of the efficient problem-specific preconditioner, (ii) customized CUDA
kernel for matrix-vector multiplication, and (iii) CUDA Graph implementation on the GPU. These advances
allow us to simulate the 𝑈(1) Dirac spin liquid state with unprecedentedly large system sizes, which is up to
𝑁𝜏 × 𝐿 × 𝐿 = 660 × 66 × 66, and reveal its novel properties. With these technical improvements, we see
the asymptotic convergence in the scaling dimensions of various fermion bilinear operators and the conserved
current operator when approaching the thermodynamic limit. The scaling dimensions find good agreement with
field-theoretical expectation, which provides supporting evidence for the conformal nature of the 𝑈(1) Dirac
spin liquid state in the QED3. Our technical advancements open an avenue to study the Dirac spin liquid state
and its transition towards symmetry-breaking phases at larger system sizes and with less computational burden.

## Cite
```
@article{feng2025scalable,
  title={Scalable Hybrid quantum Monte Carlo simulation of U (1) gauge field coupled to fermions on GPU},
  author={Feng, Kexin and Chen, Chuang and Meng, Zi Yang},
  journal={arXiv preprint arXiv:2508.16298},
  year={2025}
}
```
