import torch

def initialize_coupling_mat(Lx, Ly, Ltau, J,  K=1):
    # delta_tau = 1/J
    
    # # Warning:
    # J = J * delta_tau**2

    Lpol = 2
    dim = Ltau * Ly * Lx * Lpol
    A = torch.zeros(dim, dim)

    unit = torch.zeros(Ly*Lx*Lpol + 2, Ly*Lx*Lpol + 2)
    
    # off-diag
    # [phi^x_ij, phi^y_ij, phi^y_{i+1, j}, phi^x_{i, j+1}, phi^x_{ij, tau+1}, phi^y_{ij, tau+1}]
    # [0,        1,        Lpol+1,         Lx*Lpol       , Ly*Lx*Lpol,        Ly*Lx*Lpol + 1]
    unit[1, 0] = -K
    unit[Lpol+1, 0] = K
    unit[Lx*Lpol,  0] = -K

    unit[Lpol+1, 1] = -K
    unit[Lx*Lpol,  1] = K

    unit[Lx*Lpol, Lpol+1] = -K

    unit[Ly*Lx*Lpol, 0] = -J
    unit[Ly*Lx*Lpol+1, 1] = -J
    
    # symmetrize off-diag
    unit = unit + unit.T

    # diag
    unit[0, 0] = K+ J
    unit[1, 1] = K+ J
    unit[Lpol+1, Lpol+1] = K
    unit[Lx*Lpol, Lx*Lpol] = K
    unit[Ly*Lx*Lpol, Ly*Lx*Lpol] = J
    unit[Ly*Lx*Lpol+1, Ly*Lx*Lpol+1] = J

    # # translate / summation over i,j,tau
    # for i in range(0, Ltau*Ly*Lx*Lpol, Lpol):
    #     if i >= (Ltau-1)*Ly*Lx*Lpol - 2:
    #         break
    #     A[i: i + Ly*Lx*Lpol + 2, i: i + Ly*Lx*Lpol + 2] += unit

    # # periodic boundary
    # for i in range((Ltau-1)*Ly*Lx*Lpol - 2, Ltau*Ly*Lx*Lpol, Lpol):
    #     # for x in range(unit.size(0)):
    #     #     for y in range(unit.size(1)):
    #     #         A[(i + x) % dim, (i + y) % dim] += unit[x, y]
    #     x_idx, y_idx = torch.meshgrid(
    #         torch.arange(unit.size(0)), torch.arange(unit.size(1)), indexing='ij'
    #     )
    #     A[(i + x_idx) % dim, (i + y_idx) % dim] += unit[x_idx, y_idx]

    # # 
    # A = A.view(Ltau, Ly, Lx, Lpol, Ltau, Ly, Lx, Lpol)
    # A = A.permute([3, 2, 1, 0, 7, 6, 5, 4])

    # benchmark
    A_ben = torch.zeros(Ltau, Ly, Lx, Lpol, Ltau, Ly, Lx, Lpol)
    for tau in range(Ltau):
        for y in range(Ly):
            for x in range(Lx):
                # for a in range(2):

                ius = torch.arange(unit.size(0))
                jus = torch.arange(unit.size(1))
                x_idx, y_idx = torch.meshgrid(ius, jus, indexing='ij')
                x_idx = x_idx.reshape(-1)
                y_idx = y_idx.reshape(-1)

                shape = (Ltau, Ly, Lx, 2)
                idtau, idy, idx, ida = torch.unravel_index(x_idx, shape)
                jdtau, jdy, jdx, jda = torch.unravel_index(y_idx, shape)

                A_ben[(tau+idtau)%Ltau, (y+idy)%Ly, (x+idx)%Lx, ida, (tau+jdtau)%Ltau, (y+jdy)%Ly, (x+jdx)%Lx, jda] += unit[x_idx, y_idx]

    A_ben = A_ben.permute([3, 2, 1, 0, 7, 6, 5, 4])

    # torch.testing.assert_close(A, A_ben)

    return A_ben, unit


  