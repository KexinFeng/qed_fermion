mkdir -p report

J_array=$(echo '0.5')

Nstep=10000
bs=5
# for J in $J_array; do
#         #
#         config=$(echo local_J_${J}_Nstep_${Nstep}_bs_${bs})
#         echo $config
#         export J Nstep bs
#         #
#         sbatch --job-name=${config}_hmc \
#         --time=0-0:30:00 \
#         --qos=debug \
#         --mem-per-cpu=6G \
#         s_local.cmd

# done

sbatch --job-name=test_cond_num \
        --time=0-0:30:00 \
        --qos=debug \
        --mem-per-cpu=10G \
        s_cg.cmd
