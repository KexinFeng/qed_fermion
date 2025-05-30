mkdir -p report_bench

J_array=$(echo '1.0')
L_array=$(echo '18 22 24')

J_array=$(echo '1.0')
L_array=$(echo '16 20')

J_array=$(echo '1.0')
L_array=$(echo '20')  # 10 h (-2)

export debug=1
export asym=2
export cuda_graph=1

Nstep=1000
for L in $L_array; do
        #
        for J in $J_array; do
                #
                config=$(echo J_${J}_L_${L}_Nstep_${Nstep})
                echo $config
                export J Nstep L
                #
                sbatch --job-name=${config}_hmc \
                --time=0-0:30:00 \
                --qos=debug \
                --mem-per-cpu=12G \
                s_hmc.cmd
                # {L: size_gb} = {12: 20, 16: 30}
        done
done

