# Tunining Note

##delta_t and mass
```python
    # self.delta_t = 0.001
    # self.delta_t = 0.05
    # self.delta_t = 0.1
    # self.delta_t = 0.4

    # m = 2, T(~sqrt(m/k)) = 1.5, N = T / 0.05 = 30
    # m = 1/2, T = 0.75, N = T / 0.05 = 15
    # T/4 = 0.375

    # total_t = 0.375
    # self.N_leapfrog = int(total_t // self.delta_t)
    # 1st exp: t = 1, delta_t = 0.05, N_leapfrog = 20

    # self.N_leapfrog = 15
    # Fixing the total number of leapfrog step, then the larger delta_t, the longer time the Hamiltonian dynamic will reach, the less correlated is the proposed config to the initial config, where the correlation is in the sense that, in the small delta_t limit, almost all accpeted and p being stochastic, then the larger the total_t, the less autocorrelation. But larger delta_t increases the error amp and decreases the acceptance rate.

    # Increasing m, say by 4, the sigma(p) increases by 2. omega = sqrt(k/m) slows down by 2 [cos(wt) ~ 1 - 1/2 * k/m * t^2]. The S amplitude is not affected (since it's decided by initial cond.), but somehow H amplitude decreases by 4, similar to omega^2 decreases by 4. 
```

```