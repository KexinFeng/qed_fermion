mkdir -p report


J_array=$(echo '0.5')

Nstep=20

for J in $J_array; do
        #
        echo "Nstep_${Nstep}_J_${J}"
        export J Nstep
        #
        sbatch --job-name=hmc_sampler_Nstep_${Nstep}_J_${J} \
        --time=0-00:30:00 \
        --qos=debug \
        --mem-per-cpu=6G \
        s.cmd
done
