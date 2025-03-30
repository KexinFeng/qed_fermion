mkdir -p report


# J_array=$(echo '3 1 0.5')
J_array=$(echo '10')

# Nstep=10000
# for J in $J_array; do
#         #
#         config=$(echo hmc_J_${J}_Nstep_${Nstep})
#         echo $config
#         export J Nstep
#         #
#         sbatch --job-name=${config}_hmc \
#         --time=0-4:00:00 \
#         --qos=normal \
#         --mem-per-cpu=6G \
#         s_hmc.cmd
# done

Nstep=100
bs=5
for J in $J_array; do
        #
        config=$(echo local_J_${J}_Nstep_${Nstep}_bs_${bs})
        echo $config
        export J Nstep bs
        #
        sbatch --job-name=${config}_hmc \
        --time=0-20:00:00 \
        --qos=gpu \
        --mem-per-cpu=50G \
        s_local.cmd
done

# qos=normal

# sbatch --job-name=cond_num \
#         --time=0-20:00:00 \
#         --qos=gpu \
#         --mem-per-cpu=10G \
#         s_cg.cmd