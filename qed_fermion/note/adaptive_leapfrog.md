# Adaptive optimization
```
        # Leapfrog
        self.debug_pde = False
        self.m = 1/2 * 4 / scale * 0.05
        # self.m = 1/2

        # self.delta_t = 0.05
        # self.delta_t = 0.2/4
        # self.delta_t = 0.005
        self.delta_t = 0.04/2
        # self.delta_t = 0.0066
        # self.delta_t = 0.04/4
        self.delta_t = 0.05/4
        self.delta_t = 0.14/8
        # self.delta_t = 0.0175 = 0.07/4
        self.delta_t = 0.025  # >=0.03 will trap the leapfrog at the beginning
        # self.delta_t = 0.1 # This will be too large and trigger H0,Hfin not equal, even though N_leapfrog is cut half to 3
        # For the same total_t, the larger N_leapfrog, the smaller error and higher acceptance.
        # So for a given total_t, there is an optimal N_leapfrog which is the smallest N_leapfrog s.t. the acc is larger than say 0.9 the saturate accp (which is 1).
        # Then increasing total_t will increase N_leapfrog*, is total_t reasonable.
        # So proper total_t is a function of N_leapfrog* for a given threshold like 0.9.
        # So natually an adaptive optimization algorithm can be obtained: for a fixed N_leapfrog, check the acceptance_rate / acc_threshold and adjust total_t.

        # self.N_leapfrog = 6 # already oscilates back
        # self.N_leapfrog = 8
        # self.N_leapfrog = 2
        # self.N_leapfrog = 6
        self.N_leapfrog = 4
        # self.N_leapfrog = 8

        # CG
        # self.cg_rtol = 1e-7
        # self.max_iter = 400  # at around 450 rtol is so small that becomes nan 
        self.cg_rtol = 1e-4
        self.max_iter = 2000  # at around 450 rtol is so small that becomes nan 
        self.precon = None
        self.plt_cg = False
        self.verbose_cg = False
```