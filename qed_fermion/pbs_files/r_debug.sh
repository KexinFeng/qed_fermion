cd "$(dirname "$0")"

J_array=$(echo '1.0')
L_array=$(echo '18 22 24')

J_array=$(echo '1.0')
L_array=$(echo '16 20')

J_array=$(echo '1.25')
L_array=$(echo '66 60 56 50')  # Nleap3_taublk2_bs2_2000_8h
L_array=$(echo '80 76 70 66')  # Nleap3_taublk2_bs2_2000_8h

export debug=0
export cuda_graph=1
export bs=1

export suffix=debug
export asym=1
export compact=0
export K=0
export dtau=0.1
export precon=1

export compute_BB=0
export compute_spsm=0

export seed=250

Nstep=10000
for L in $L_array; do
        #
        for J in $J_array; do
                #
                config=$(echo L${L}a${asym}J${J}K${K})
                echo $config
                export J Nstep L
                #
                sbatch --job-name=${config} \
                --time=0-0:10:00 \
                --qos=debug \
                --mem-per-cpu=6G \
                s_hmc_debug.cmd
                # {L: size_gb} = {12: 20, 16: 30}
        done
done

