cd "$(dirname "$0")"

export debug=0
export cuda_graph=0
export mxitr=400  # 1000
export Nrv=100

# export Lx=6
# export Ltau=$((10 * Lx))
# python load_se.py

# export Lx=8
# export Ltau=$((10 * Lx))
# python load_se.py

export Lx=10
export Ltau=$((10 * Lx))
python load_se.py



