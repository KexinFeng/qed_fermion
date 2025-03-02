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
    return coeff * (xk.abs()**2 * lmd).sum().view(-1) / L


def action_tau(x, Ltau):
    s = 0
    for tau in range(Ltau):
        s += 1/2 * (x[..., (tau+1)%Ltau] - x[..., tau])**2
    return s.sum().view(-1)


xt = torch.tensor([[ 0, 1, 2, 3, 5]], dtype=torch.float)

print(torch.fft.fft(xt, dim=1))

Ltau = xt.view(-1).size(0)

print(action_boson_tau(xt, Ltau))
print(action_tau(xt, Ltau))
print(1/2 * xt @ get_A(Ltau) @ xt.T)

torch.testing.assert_close(action_boson_tau(xt, Ltau), action_tau(xt, Ltau))
torch.testing.assert_close((1/2 * xt @ get_A(Ltau) @ xt.T).view(-1), action_tau(xt, Ltau))
print('=====')


k = torch.fft.fftfreq(Ltau)
F = torch.fft.fft(torch.eye(Ltau), dim=0)  
print(F @ F.T.conj() / Ltau)
A = get_A(Ltau).to(torch.complex64)
Ak2 = 1/Ltau * F.T.conj() @ A @ F
print(f'Ak2 = {torch.real(torch.diag(Ak2))}')
Ak2 = torch.real(torch.diag(Ak2))

lmd = 2 * (1 - torch.cos(k * 2*torch.pi))
print(f'lmd={lmd}')

torch.testing.assert_close(lmd, Ak2)

print('------')




