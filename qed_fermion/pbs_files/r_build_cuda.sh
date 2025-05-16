mkdir -p report

J_array=$(echo '1.0')
L_array=$(echo '6')

Nstep=6000
for L in $L_array; do
        #
        for J in $J_array; do
                #
                config=$(echo build_cuda)
                echo $config
                export J Nstep L
                #
                sbatch --job-name=${config}_hmc \
                --time=0-0:20:00 \
                --qos=debug \
                --mem-per-cpu=16G \
                s_build_cuda.cmd
                # {L: size_gb} = {12: 20, 16: 30}
        done
done

# sbatch --job-name=test_cond_num \
#         --time=0-0:30:00 \
#         --qos=debug \
#         --mem-per-cpu=10G \
#         s_cg.cmd
