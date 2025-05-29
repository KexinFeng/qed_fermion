cd /users/4/fengx463/hmc/qed_fermion/qed_fermion/post_processors/6x60_8x80_10x100_bs2/benchmark_se/
export cuda_graph=1

export Lx=6
export Ltau=60
python3 load_se.py

export Lx=8
export Ltau=80
python3 load_se.py

export Lx=10
export Ltau=100
python3 load_se.py

## Plot
# export Lx=6
# export Ltau=60
# python3 plot_local_hmc_energy_J_self_dqmc.py
# python3 plot_local_hmc_energy_J_dqmc_vs_hmc.py
# python3 plot_spsm_dqmc_vs_hmc.py

# export Lx=8
# export Ltau=80
# python3 plot_local_hmc_energy_J_self_dqmc.py
# python3 plot_local_hmc_energy_J_dqmc_vs_hmc.py
# python3 plot_spsm_dqmc_vs_hmc.py

# export Lx=10
# export Ltau=100
# python3 plot_local_hmc_energy_J_self_dqmc.py
# python3 plot_local_hmc_energy_J_dqmc_vs_hmc.py
# python3 plot_spsm_dqmc_vs_hmc.py
