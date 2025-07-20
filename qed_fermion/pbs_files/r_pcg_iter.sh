cd "$(dirname "$0")"

# J_array=$(echo '1.0 1.5 2.0 2.3 2.5 3.0')
# J_array=$(echo '1.25')
# L_array=$(echo '36 30 20')  # 10 h (-2)
# L_array=$(echo '16 12 10')  # 10 h (-2)

# spsm_r lattice sizes
# L_array=$(echo '46 40 36 30 20 16 12 10 8 6')
# L_array=$(echo '56 50')
# L_array=$(echo '60 66')
L_array=$(echo '36 30 20 16 12 10 8 6')
L_array=$(echo '46 40')
L_array=$(echo '50')
L_array=$(echo '30 20 16 12 10 8 6')
L_array=$(echo '40 36')
L_array=$(echo '50 46')

# BB_r lattice sizes
L_array=$(echo '36 30 20 16 12 10 8 6')
L_array=$(echo '46 40')
L_array=$(echo '60 56 50')

# L_array=$(echo '60 56 50 46 40 36 30 20 16 12 10 8 6')
# L_array=$(echo '60 56 50 46 40 36 30 26 20 16 10')
L_array=$(echo '60 50 40 30 20 10')
L_array=$(echo '60 40 20')
max_iter_array=$(echo '200 400 800 1200')

# J_array=$(echo '1.0')
# L_array=$(echo '6 8 10')  # 10 h (-2)

Nstep=6000
export debug=0
export cuda_graph=1
export bs=1

export suffix=pcg_iter
export asym=1
export compact=0
export K=0
export dtau=0.1
export precon=1

export compute_BB=0
export compute_spsm=0

export J=1.25

export seed=49

for L in $L_array; do
        #
        for max_iter in $max_iter_array; do
                #
                config=$(echo mitr${max_iter}_L${L})
                echo $config
                export L max_iter
                #
                sbatch --job-name=${config} \
                --time=6-23:59:00 \
                --qos=gpu \
                --mem-per-cpu=8G \
                s_hmc_noncmp.cmd
        done
done


# srun --pty --partition=interactive --qos=ood --mem=1G --cpus-per-task=1 --time=01:00:00 bash -i

