self.m = 1/2 * 4
self.delta_t = 0.05
# m = 2, T(~sqrt(m/k)) = 1.5, N = T / 0.05 = 30
# m = 1/2, T = 0.75, N = T / 0.05 = 15
# T/4 = 0.375

# total_t = 0.375
# self.N_leapfrog = int(total_t // self.delta_t)
# 1st exp: t = 1, delta_t = 0.05, N_leapfrog = 20

self.N_leapfrog = 15

N_leapfrog is at pi phase, i.e. 1/2 T, the final state is far enough away from intial state.

Such behaviour is seen in larger sizes up to Lx=Ly=30.
