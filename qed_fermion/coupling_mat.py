import torch

def initialize_coupling_mat(Lx, Ly, Ltau, J,  K=1, delta_tau=1):
    A = torch.zeros(2, Lx, Ly, Ltau, 2, Lx, Ly, Ltau)

    pol1, x1, y1, tau1, pol2, x2, y2, tau2 = [], [], [], [], [], [], [], []
    for pol in range(2):
        for x in range(Lx):
            for y in range(Ly):
                for tau in range(Ltau):
                    # phi(r)**2
                    pol1.append(pol)
                    x1.append(x)
                    y1.append(y)
                    tau1.append(tau)

                    pol2.append(pol)
                    x2.append(x)
                    y2.append(y)
                    tau2.append(tau)

                    # phi(r + e_tau)**2
                    pol1.append(pol)
                    x1.append(x)
                    y1.append(y)
                    tau1.append((tau+1) % Ltau)

                    pol2.append(pol)
                    x2.append(x)
                    y2.append(y)
                    tau2.append((tau+1) % Ltau)

    # A[pol1, x1, y1, tau1, pol2, x2, y2, tau2] = 0.5/J/delta_tau
    for ppol1, xx1, yy1, ttau1, ppol2, xx2, yy2, ttau2 in zip(pol1, x1, y1, tau1, pol2, x2, y2, tau2): 
        A[ppol1, xx1, yy1, ttau1, ppol2, xx2, yy2, ttau2] += 0.5/J/delta_tau

    pol1, x1, y1, tau1, pol2, x2, y2, tau2 = [], [], [], [], [], [], [], []
    for pol in range(2):
        for x in range(Lx):
            for y in range(Ly):
                for tau in range(Ltau):
                    # phi(r)phi(r + etau)
                    pol1.append(pol)
                    x1.append(x)
                    y1.append(y)
                    tau1.append(tau)

                    pol2.append(pol)
                    x2.append(x)
                    y2.append(y)
                    tau2.append((tau + 1) % Ltau)

                    # phi(r + e_tau)phi(r)
                    pol1.append(pol)
                    x1.append(x)
                    y1.append(y)
                    tau1.append((tau+1) % Ltau)

                    pol2.append(pol)
                    x2.append(x)
                    y2.append(y)
                    tau2.append(tau)  

    # A[pol1, x1, y1, tau1, pol2, x2, y2, tau2] = -0.5/J/delta_tau
    for ppol1, xx1, yy1, ttau1, ppol2, xx2, yy2, ttau2 in zip(pol1, x1, y1, tau1, pol2, x2, y2, tau2): 
        A[ppol1, xx1, yy1, ttau1, ppol2, xx2, yy2, ttau2] += -0.5/J/delta_tau

    # ------ K terms ------
    pol1, x1, y1, tau1, pol2, x2, y2, tau2 = [], [], [], [], [], [], [], []
    for x in range(Lx):
        for y in range(Ly):
            for tau in range(Ltau): 
                idxs = [(0, x, y, tau), (1, x, y, tau), (0, x, (y+1) % Ly, tau), (1, (x+1) % Lx, y, tau)]
                # diagonal
                for lft, rgt in zip(idxs, idxs):
                    pol_lft, x_lft, y_lft, tau_lft = lft
                    pol_rgt, x_rgt, y_rgt, tau_rgt = rgt
                    pol1.append(pol_lft)
                    pol2.append(pol_rgt)
                    x1.append(x_lft)
                    x2.append(x_rgt)
                    y1.append(y_lft)
                    y2.append(y_rgt)
                    tau1.append(tau_lft)
                    tau2.append(tau_rgt)
                
                # (2, 3)
                pol1.extend([1, 0])
                x1.extend([x, x])
                y1.extend([y, (y+1)%Ly])
                tau1.extend([tau, tau])

                pol2.extend([0, 1])
                x2.extend([x, x])
                y2.extend([(y+1)%Ly, y])
                tau2.extend([tau, tau])

                # (1, 4)
                pol1.extend([0, 1])
                x1.extend([x, (x+1)%Lx])
                y1.extend([y, y])
                tau1.extend([tau, tau])

                pol2.extend([1, 0])
                x2.extend([(x+1)%Lx, x])
                y2.extend([y, y])
                tau2.extend([tau, tau])

    # A[pol1, x1, y1, tau1, pol2, x2, y2, tau2] = 0.5 * K * delta_tau
    for ppol1, xx1, yy1, ttau1, ppol2, xx2, yy2, ttau2 in zip(pol1, x1, y1, tau1, pol2, x2, y2, tau2): 
        A[ppol1, xx1, yy1, ttau1, ppol2, xx2, yy2, ttau2] += 0.5 * K * delta_tau


    pol1, x1, y1, tau1, pol2, x2, y2, tau2 = [], [], [], [], [], [], [], []
    for x in range(Lx):
        for y in range(Ly):
            for tau in range(Ltau): 
                # (0, x,        y, tau), 
                # (1, x,        y, tau), 
                # (0, x,        (y+1) % Ly, tau), 
                # (1, (x+1) % Lx, y, tau)
                idxs = [(0, x, y, tau), (1, x, y, tau), (0, x, (y+1) % Ly, tau), (1, (x+1) % Lx, y, tau)]
                for iidx, iidy in [(1, 2), (1, 3), (2, 4), (3, 4)]:
                    iidx -= 1
                    iidy -= 1
                    pol1.extend([idxs[iidx][0], idxs[iidy][0]])
                    pol2.extend([idxs[iidy][0], idxs[iidx][0]])
                    x1.extend([idxs[iidx][1], idxs[iidy][1]])
                    x2.extend([idxs[iidy][1], idxs[iidx][1]])
                    y1.extend([idxs[iidx][2], idxs[iidy][2]])
                    y2.extend([idxs[iidy][2], idxs[iidx][2]])
                    tau1.extend([idxs[iidx][3], idxs[iidy][3]])
                    tau2.extend([idxs[iidy][3], idxs[iidx][3]])
                    
    # A[pol1, x1, y1, tau1, pol2, x2, y2, tau2] = -0.5* K * delta_tau
    for ppol1, xx1, yy1, ttau1, ppol2, xx2, yy2, ttau2 in zip(pol1, x1, y1, tau1, pol2, x2, y2, tau2): 
        A[ppol1, xx1, yy1, ttau1, ppol2, xx2, yy2, ttau2] += -0.5* K * delta_tau

    return A
