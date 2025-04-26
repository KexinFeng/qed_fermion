mkdir -p report


J_array=$(echo '1.0 1.5 2.0 2.5 3.0 3.5')
L_array=$(echo '6 8 10')  # 10 h (-2)

J_array=$(echo '1.0')
L_array=$(echo '12 14 16')  # 16 h (-2)

J_array=$(echo '1.0')
L_array=$(echo '18 20 22 24') # 28 h (-2)

Nstep=6000
for L in $L_array; do
        #
        for J in $J_array; do
                #
                config=$(echo J${J}L${L}Nstep${Nstep})
                echo $config
                export J Nstep L
                #
                sbatch --job-name=${config}_hmc \
                --time=0-30:00:00 \
                --qos=gpu \
                --mem-per-cpu=4G \
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

