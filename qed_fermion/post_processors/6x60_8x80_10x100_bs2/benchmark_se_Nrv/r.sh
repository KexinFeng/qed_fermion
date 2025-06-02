mkdir -p report_se


Nrv_array=$(echo '200 100 50 10')
L_array=$(echo '6 8 10')  # 10 h (-2)

export debug=0
export cuda_graph=1
export mxitr=400  # 1000

for Nrv in $Nrv_array; do
        #
        for Lx in $L_array; do
                #
                config=$(echo Lx${Lx}Nrv${Nrv}mxitr${mxitr})
                echo $config
                export Nrv Lx
                export Ltau=$((10 * Lx))
                sbatch --job-name=${config} \
                        --time=0-3:00:00 \
                        --qos=gpu \
                        --mem-per-cpu=18G \
                        s_hmc.cmd
        done
done

# sbatch --job-name=se \
#         --time=0-3:00:00 \
#         --qos=gpu \
#         --mem-per-cpu=10G \
#         s_hmc.cmd
   
