cd "$(dirname "$0")"

export Lx=6
export Ltau=60
python3 load_write2file_convert.py

export Lx=8
export Ltau=80
python3 load_write2file_convert.py

# export Lx=10
# export Ltau=100
# python3 load_write2file_convert.py




