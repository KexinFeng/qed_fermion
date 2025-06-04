mkdir -p report_precon

# J_array=$(echo '1.0')
# L_array=$(echo '26 28 30')
J_array=$(echo '1.0')
L_array=$(echo '6 8 10')  # 10 h (-2)
J_array=$(echo '1.0')
L_array=$(echo '12 14 16 18')  # 10 h (-2)
J_array=$(echo '1.0')
L_array=$(echo '20 22 24 26')  # 10 h (-2)
L_array=$(echo '24')  # 10 h (-2)

# L_array=$(echo '30 40 50 60 36 46 56 66')  # 500G
L_array=$(echo '20 16 12 10 8 6')  # 50G
L_array=$(echo '66 60 56 50 46 40')  # 500G
L_array=$(echo '36 30 20 16 12 10')  # 500G
L_array=$(echo '40')  # 500G
# L_array=$(echo '24 20')  # 500G

export debug=0
export asym=1
export cuda_graph=0

for L in $L_array; do
        #
        for J in $J_array; do
                #
		config=$(echo L${L}a${asym})
                echo $config
                export L
                #
        	sbatch --job-name=${config}_hmc \
                --time=0-20:00:00 \
                --qos=hugemem \
                --mem=1500G \
                s_hmc_precon.cmd
	done
done

# for L in $L_array; do
#         #
#         for J in $J_array; do
#                 #
# 		config=$(echo L_${L})
#                 echo $config
#                 export L
#                 #
#         	sbatch --job-name=${config}_hmc \
#                 --time=0-15:00:00 \
#                 --qos=hugemem \
#                 --mem=2000G \
#                 s_hmc_precon.cmd
# 	done
# done

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
