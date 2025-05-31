mkdir -p report_bench

J_array=$(echo '1.0')
L_array=$(echo '18 22 24')

J_array=$(echo '1.0')
L_array=$(echo '16 20')

J_array=$(echo '1.0')
L_array=$(echo '36')  # 10 h (-2)

export debug=1
export asym=2
export cuda_graph=1

Nstep=10
mem=6
for L in $L_array; do
        #
        for J in $J_array; do
                #
                config=$(echo m${mem}_L_${L}_Nstep_${Nstep})
                echo $config
                export J Nstep L
                #
                sbatch --job-name=${config} \
                --time=0-0:30:00 \
                --qos=debug \
                --mem-per-cpu=${mem}G \
                s_hmc.cmd
                # {L: size_gb} = {12: 20, 16: 30}
        done
done

