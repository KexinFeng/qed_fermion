mkdir -p report_bench

# J_array=$(echo '1.0')
# L_array=$(echo '4 6 8 10')  # 10 h (-2)

J_array=$(echo '1.0 1.5 2.0 2.3 2.5 3.0')
L_array=$(echo '6 8 10')  # 10 h (-2)

# J_array=$(echo '1.0 1.5 2.0 2.5 3.0 3.5')
# L_array=$(echo '12 14 16')  # 16 h (-2)

# J_array=$(echo '1.0')
# L_array=$(echo '18 20 22 24') # 26 h (-2)  16: 8g HBM, 20: 15g HBM

# J_array=$(echo '1.0')
# L_array=$(echo '24') # 32 h 8g RAM 26g HBM

# J_array=$(echo '1.0 1.5 2.0 2.5 3.0')
# L_array=$(echo '8')  # 10 h (-2)

# J_array=$(echo '1.5')
# L_array=$(echo '6')  # 10 h (-2)

Nstep=5000

export debug=0
export asym=2
export cuda_graph=1

for L in $L_array; do
        #
        for J in $J_array; do
                #
                config=$(echo J${J}L${L}a${asym})
                echo $config
                export J Nstep L
                #
                sbatch --job-name=${config} \
                --time=0-30:00:00 \
                --qos=gpu \
                --mem-per-cpu=12G \
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

