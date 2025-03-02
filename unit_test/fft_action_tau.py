import torch

device = 'cpu'

def get_A(L):
    A = torch.zeros(L, L)
    for tau in range(L):
        A[tau, (tau-1)%L] = -1
        A[tau, tau] = 2
        A[tau, (tau+1)%L] = -1
    return A

def action_boson_tau(x, Ltau):
    """
    x:  [bs, 2, Lx, Ly, Ltau]
    """
    coeff = 1/2 
    xk = torch.fft.fft(x, dim=-1)
    L = Ltau
    k = torch.fft.fftfreq(L).to(device)    
    lmd = 2 * (1 - torch.cos(k * 2*torch.pi))

    # F = torch.fft.fft(torch.eye(3), dim=0)
    # torch.eye(3) = 1/L * F.conj().T @ F
    return coeff * (xk.abs()**2 * lmd).sum() / L


def action_tau(x, Ltau):
    s = 0
    for tau in range(Ltau):
        s += 1/2 * (x[..., (tau+1)%Ltau] - x[..., tau])**2
    return s.sum()


xt = torch.tensor([[ 0.5606,  0.5551,  0.7350,  0.6990,  0.9548,  0.9235]])
xt = torch.tensor([[ 0, 1, 2, 3, 5]], dtype=torch.float)

print(torch.fft.fft(xt, dim=1))

Ltau = xt.view(-1).size(0)

print(action_boson_tau(xt, Ltau))
print(action_tau(xt, Ltau))
print(1/2 * xt @ get_A(Ltau) @ xt.T)
print('=====')


k = torch.fft.fftfreq(Ltau)
F = torch.fft.fft(torch.eye(Ltau), dim=0)  
print(F @ F.T.conj() / Ltau)
A = get_A(Ltau).to(torch.complex64)
Ak2 = 1/Ltau * F.T.conj() @ A @ F
print(f'Ak2 = {torch.real(torch.diag(Ak2))}')

lmd = 2 * (1 - torch.cos(k * 2*torch.pi))
print(f'lmd={lmd}')

print('------')




