mkdir -p report_se

#
sbatch --job-name=se \
        --time=0-3:00:00 \
        --qos=gpu \
        --mem-per-cpu=16G \
        s_hmc.cmd
   
