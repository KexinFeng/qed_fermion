import torch
import matplotlib.pyplot as plt
plt.ion()
from matplotlib import rcParams
import os
rcParams['figure.raise_window'] = False

script_path = os.path.dirname(os.path.abspath(__file__))


Lx = 10
Ltau = 100
debug = False

def get_precon_man(Lx, Ltau):
    # Lx = int(os.getenv("L", '6'))
    # asym = float(os.environ.get("asym", '1'))
    # Ltau = int(asym*Lx * 10) 
    
    Ly = Lx
    vs = Lx * Ly
    file_path = script_path + f"/../preconditioners/precon_ckpt_L_{Lx}_Ltau_{Ltau}_dtau_0.1_t_1.pt"

    size = vs * Ltau

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cdtype = torch.complex64  

    # indices # [2, nnz]
    # values # [nnz]

    out_values = []
    out_indices = []

    ## Rank 0
    # Edge values
    edge_values = [
        0.91698223, 1.02107012, 1.1740706,  1.24646831, 1.28092396, 1.29741096,
        1.30534041, 1.30917168, 1.3110323,  1.31193936
    ]

    # Tail values
    tail_values = [
        1.31167626, 1.31074941, 1.30885756, 1.30498219, 1.29699361, 1.28042281,
        1.24588311, 1.17343438, 1.02052784
    ]

    # Number of edge and tail values
    n_edge = len(edge_values)
    n_tail = len(tail_values)

    # Number of middle values to fill
    n_middle = Ltau - n_edge - n_tail

    # Repeat the middle value
    middle_value = 1.312381281
    middle_values = torch.full((n_middle,), middle_value, dtype=cdtype, device=device)

    values = torch.cat([
        torch.tensor(edge_values, dtype=cdtype, device=device),
        middle_values,
        torch.tensor(tail_values, dtype=cdtype, device=device)
    ])
    # Repeat each value in 'values' vs times and interleave them
    values = values.repeat_interleave(vs)
    indices = torch.arange(vs * Ltau, device=device)
    indices = torch.stack([indices, indices], dim=0)

    out_values.append(values)
    out_indices.append(indices)


    ## Rank 1
    # Edge values
    edge_values_list = [[
    0.475227, 0.69804072,  0.80315554,  0.85303694,  0.87684089,
    0.88826168, 0.89376724, 0.89643389, 0.89773124, 0.89836514,
    0.89867532, 0.89882821
    ]]
    # Tail values
    tail_values_list = [[
    0.89785033, 0.89652652, 0.89382064, 0.88826007,
    0.87676316, 0.85285789, 0.80287397, 0.69771057
    ]]
    middle_value_list = [0.8989771]

    ## Rank 2
    edge_values_list.append([0.32525402, 0.47824016, 0.550632, 0.58508581, 0.60157132,
        0.60949975, 0.61333305, 0.61519265])
    tail_values_list.append([0.61507541, 0.61318511, 0.60930747, 0.60131705,
        0.58475631, 0.5502165, 0.47778079])
    middle_value_list.append(0.616974)

    ## Rank 3
    edge_values_list.append([0.22288805,    0.32804558,    0.3779504,     0.40176818,    0.41319385,
        0.4187026,     0.42137364,    0.42267111])
    tail_values_list.append([0.42132217,    0.41861743,    0.41305307,    0.40155935,
        0.37766892,    0.3276957])
    middle_value_list.append( 0.42391914)

    ## Rank 4
    edge_values_list.append([0.15300766,    0.22541766,    0.25988016,    0.27637246,    0.28430402,
        0.28813654,    0.29000109,    0.29090756,    0.291352  ])
    tail_values_list.append([0.29132634,    0.29086968,    0.28994057,    0.28805143,    0.28417516,
        0.27618375,    0.25962788,    0.22511749])
    middle_value_list.append(0.29178312)

    ## Rank 5
    edge_values_list.append([ 0.10518534,    0.15511395,    0.17894313,    0.190377,      0.19588688,
        0.1985613,     0.19986567,    0.20050026,    0.20081173,    0.20096488,
        0.20104057])
    tail_values_list.append([0.20103745,    0.20095888,    0.20079921,    0.20047785,    0.19982809,
        0.19850214,    0.19579838,    0.19023941,    0.17874525,    0.15486649])
    middle_value_list.append(0.20111492)

    ## Rank 6
    edge_values_list.append([ 0.10691644,    0.12341493,    0.13135427,    0.13519259,
        0.13706173,    0.13796897,    0.13841458 ])
    tail_values_list.append([0.13839549,    0.13793865,    0.13701266,
        0.13511762,    0.13124515,    0.12326149,    0.10670882])
    middle_value_list.append(0.13884714)


    for rank in range(1, 7):
        edge_values = edge_values_list[rank-1]
        tail_values = tail_values_list[rank-1]

        # Number of edge and tail values
        n_edge = len(edge_values)
        n_tail = len(tail_values)

        # Number of middle values to fill
        n_middle = Ltau - n_edge - n_tail - rank - (1 if rank == 6 else 0)

        # Repeat the middle value
        middle_value = middle_value_list[rank-1]
        middle_values = torch.full((n_middle,), middle_value, dtype=cdtype, device=device)

        values = torch.cat([
            torch.tensor(edge_values, dtype=cdtype, device=device),
            middle_values,
            torch.tensor(tail_values, dtype=cdtype, device=device)
        ])
        # Repeat each value in 'values' vs times and interleave them
        values = values.repeat_interleave(vs)
        indices = torch.arange(0 if rank < 6 else vs, vs * (Ltau - rank), device=device)
        indices = torch.stack([indices, indices + rank * vs], dim=0)

        out_values.append(values)
        out_indices.append(indices)

        out_values.append(values)
        out_indices.append(torch.flip(indices, dims=(0,)))


    corner_value_list = [-0.47505581, -0.32495788, -0.22264314, -0.15278356, -0.10501137]

    for rank in range(1, 6):
        values = torch.tensor([corner_value_list[rank-1]], dtype=cdtype, device=device)
        values = values.repeat_interleave(vs)
        indices = torch.arange(vs * 1, device=device)
        indices = torch.stack([indices, indices + (Ltau - rank) * vs], dim=0)
        out_values.append(values)
        out_indices.append(indices)

        out_values.append(values)
        out_indices.append(torch.flip(indices, dims=(0,)))
    
    precon_man = torch.sparse_coo_tensor(
        torch.cat(out_indices, dim=1),
        torch.cat(out_values, dim=0),
        size=(size, size),
        dtype=cdtype,
        device=device
    ).coalesce()
    
    # Save
    precon_dict = {"indices": precon_man.indices().cpu(),
                    "values": precon_man.values().cpu(),
                    "size": precon_man.size()
                   }
    os.makedirs(script_path + f"/../preconditioners", exist_ok=True)
    torch.save(precon_dict, file_path)
    print(f"Data saved to {file_path}")

    if debug:
        file_path = f"./qed_fermion/preconditioners/precon_ckpt_L_{Lx}_Ltau_{Ltau}_dtau_0.1_t_1.pt"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cdtype = torch.complex64  # or torch.float32, depending on your data

        precon_dict = torch.load(file_path, map_location=device)
        print(f"Loaded preconditioner from {file_path}")

        indices = precon_dict["indices"].to(device)
        values = precon_dict["values"].to(device)

        precon = torch.sparse_coo_tensor(
            indices,
            values,
            size=precon_dict["size"],
            dtype=cdtype,
            device=device
        ).coalesce()

        print("precon_man indices:", precon_man.indices())
        print("precon_man values:", precon_man.values())
        print("precon_man size:", precon_man.size())

        # Plot
        plt.figure(figsize=(8, 8))
        plt.spy(precon_man.to_dense().real, markersize=0.5)
        plt.title("Sparsity Pattern of Preconditioner")
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.show(block=False)

        dbstop = 1

        torch.testing.assert_close(precon_man.to_dense(), precon.to_dense(), rtol=1e-2, atol=3e-3)
        print("precon_man is close to precon.")

        dbstop = 1


if __name__ == '__main__':
    get_precon_man(Lx, Ltau)
    