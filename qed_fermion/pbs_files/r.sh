mkdir -p report


J_array=$(echo '3 1 0.5')

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
#         s.cmd
# done

Nstep=3000
bs=5
for J in $J_array; do
        #
        config=$(echo local_J_${J}_Nstep_${Nstep}_bs_${bs})
        echo $config
        export J Nstep bs
        #
        sbatch --job-name=${config}_hmc \
        --time=0-12:00:00 \
        --qos=gpu \
        --mem-per-cpu=6G \
        s_local.cmd
done

# qos=normal
