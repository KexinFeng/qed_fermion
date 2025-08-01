cd "$(dirname "$0")"

# J_array=$(echo '1.0')
# L_array=$(echo '4 6 8 10')  # 10 h (-2)

J_array=$(echo '1.0 1.5 2.0 2.1 2.2 2.3 2.4 2.5 3.0')
L_array=$(echo '6 8 10')  # 10 h (-2)
L_array=$(echo '10')  # 10 h (-2)

# J_array=$(echo '1.0 1.5 2.0 2.3 2.5 3.0')
# J_array=$(echo '1.25')
# L_array=$(echo '36 30 20')  # 10 h (-2)
# L_array=$(echo '16 12 10')  # 10 h (-2)
# L_array=$(echo '40 36 30 20 16 12 10')


# J_array=$(echo '1.0')
# L_array=$(echo '6 8 10')  # 10 h (-2)

Nstep=10000
export debug=0
export cuda_graph=1
export bs=2

export suffix=noncmp_small_BBr
export asym=1
export compact=0
export K=0
export dtau=0.1
export precon=1

export compute_BB=1
export compute_spsm=0

export seed=251

for L in $L_array; do
        #
        for J in $J_array; do
                #
                config=$(echo nL${L}a${asym}J${J})
                echo $config
                export J Nstep L
                #
                sbatch --job-name=${config} \
                --time=2-00:00:00 \
                --qos=gpu \
                --mem-per-cpu=8G \
                s_hmc.cmd
        done
done

# Nstep=100
# bs=5
# Ltau=200
# for J in $J_array; do
#         #
#         config=$(echo local_J_${J}_Nstep_${Nstep}_bs_${bs}_Ltau_${Ltau})
#         echo $config
#         export J Nstep bs Ltau
#         #
#         sbatch --job-name=${config}_hmc \
#         --time=0-24:00:00 \
#         --qos=gpu \
#         --mem-per-cpu=50G \
#         s_local.cmd
# done

# qos=normal

# sbatch --job-name=cg_convergence \
#         --time=2-00:00:00 \
#         --qos=normal \
#         --mem-per-cpu=100G \
#         s_cg.cmd


# srun --pty --partition=interactive --qos=ood --mem=1G --cpus-per-task=1 --time=01:00:00 bash -i

