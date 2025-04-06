mkdir -p report


J_array=$(echo '1')
L_array=$(echo '6 8 10')
J_array=$(echo '1')
L_array=$(echo '6')
Nstep=5000
for J in $J_array; do
        #
        config=$(echo hmc_J_${J}_J_${L}_Nstep_${Nstep})
        echo $config
        export J Nstep L
        #
        sbatch --job-name=${config}_hmc \
        --time=0-24:00:00 \
        --qos=gpu \
        --mem-per-cpu=10G \
        s_hmc.cmd
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

# sbatch --job-name=cond_num \
#         --time=0-20:00:00 \
#         --qos=gpu \
#         --mem-per-cpu=10G \
#         s_cg.cmd