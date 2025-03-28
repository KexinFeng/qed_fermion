function [L_indices, L_values] = ichol(M_indices, M_values, Nx, Ny)
    % Create sparse matrix O from input indices and values
    M = sparse(M_indices(1, :), M_indices(2, :), M_values, Nx, Ny);

    O = M' * M;

    disp(size(M_indices));
    disp(size(M_values));
    disp(size(O));

    % Incomplete Cholesky factorization for preconditioning
    % O ~ L*L' -> O_inv ~ L'^-1 * L^-1
    L = ichol(O, struct('diagcomp', 0.0001));

    % Extract indices and values for COO sparse storage of L
    [row_L, col_L, val_L] = find(L);
    L_indices = [row_L, col_L];
    L_values = val_L;
end