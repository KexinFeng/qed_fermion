mkdir -p report

J_array=$(echo '1')
L_array=$(echo '10')

Nstep=6000
for L in $L_array; do
        #
        for J in $J_array; do
                #
                config=$(echo J_${J}_Nstep_${Nstep})
                echo $config
                export J Nstep L
                #
                sbatch --job-name=${config}_hmc \
                --time=0-0:10:00 \
                --qos=debug \
                --mem-per-cpu=10G \
                s_hmc.cmd
        done
done

# sbatch --job-name=test_cond_num \
#         --time=0-0:30:00 \
#         --qos=debug \
#         --mem-per-cpu=10G \
#         s_cg.cmd
