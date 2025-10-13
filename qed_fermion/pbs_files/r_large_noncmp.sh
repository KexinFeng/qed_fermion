cd "$(dirname "$0")"

# J_array=$(echo '1.0')
# L_array=$(echo '4 6 8 10')  # 10 h (-2)

# J_array=$(echo '1.0 1.5 2.0 2.3 2.5 3.0')
# L_array=$(echo '6 8 10')  # 10 h (-2)

# J_array=$(echo '1.0 1.5 2.0 2.5 3.0 3.5')
# L_array=$(echo '12 14 16')  # 16 h (-2)

# J_array=$(echo '1.0')
# L_array=$(echo '18 20 22 24') # 26 h (-2)  16: 8g HBM, 20: 15g HBM

# J_array=$(echo '1.0')
# L_array=$(echo '24') # 32 h 8g RAM 26g HBM

# J_array=$(echo '1.0 1.5 2.0 2.3 2.5 3.0')
J_array=$(echo '1.25')
# L_array=$(echo '36 30 20')  # 10 h (-2)
# L_array=$(echo '16 12 10')  # 10 h (-2)

# spsm_r lattice sizes
# L_array=$(echo '46 40 36 30 20 16 12 10 8 6')
# L_array=$(echo '56 50')
# L_array=$(echo '60 66')
L_array=$(echo '36 30 20 16 12 10 8 6')
L_array=$(echo '46 40')
L_array=$(echo '50')
# L_array=$(echo '30 20 16 12 10 8 6')
# L_array=$(echo '40 36')
# L_array=$(echo '50 46')

# BB_r lattice sizes
L_array=$(echo '36 30 20 16 12 10')
# L_array=$(echo '46 40')
# L_array=$(echo '60 56 50')

# J_array=$(echo '1.0')
# L_array=$(echo '6 8 10')  # 10 h (-2)

Nstep=8000
export debug=0
export cuda_graph=1
export cuda_graph_se=1
export bs=2

export suffix=ncmpK1_spsm_tau
export asym=1
export compact=0
export K=1
export dtau=0.1
export precon=1

export compute_BB=0
export compute_spsm=0
export compute_spsm_tau=1

export seed=326

for L in $L_array; do
        #
        for J in $J_array; do
                #
                config=$(echo n${L}J${J}K${K})
                echo $config
                export J Nstep L
                #
                sbatch --job-name=${config} \
                --time=3-23:59:00 \
                --qos=gpu \
                --mem-per-cpu=8G \
                s_hmc_noncmp.cmd
        done
done


# srun --pty --partition=interactive --qos=ood --mem=1G --cpus-per-task=1 --time=01:00:00 bash -i

# """
# [fengx463@hpc2021-io2 ~]$ /home/fengx463/hmc/qed_fermion/qed_fermion/pbs_files/r_large_noncmp.sh
# n60J1.25K0
# Submitted batch job 2575259
# n56J1.25K0
# Submitted batch job 2575260
# n50J1.25K0
# sbatch: error: QOSMaxSubmitJobPerUserLimit
# sbatch: error: Batch job submission failed: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)

# K1 60 56 50 haven't been submitted
# """
