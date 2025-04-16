mkdir -p report

J_array=$(echo '1.0')
L_array=$(echo '18 20 22 24')


for L in $L_array; do
        #
        for J in $J_array; do
                #
		config=$(echo L_${L})
                echo $config
                export L
                #
        	sbatch --job-name=${config}_hmc --time=0-10:00:00 --qos=normal --mem=500G s_hmc.cmd
	done
done

# Nstep=100
# bs=5
# Ltau=200
# for J in $J_array; do
#         #
#         config=$(echo local_J_${J}_Nstep_${Nstep}_bs_${bs}_Ltau_${Ltau})
#         echo $config
#         export J Nstep bs Ltau
#         #
#         sbatch --job-name=${config}_hmc \
#         --time=0-24:00:00 \
#         --qos=gpu \
#         --mem-per-cpu=50G \
#         s_local.cmd
# done

# qos=normal

# sbatch --job-name=cg_convergence \
#         --time=2-00:00:00 \
#         --qos=normal \
#         --mem-per-cpu=100G \
#         s_cg.cmd
