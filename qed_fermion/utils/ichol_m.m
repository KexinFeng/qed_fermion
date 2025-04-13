function [O_inv_indices_i, O_inv_indices_j, O_inv_values] = preconditioner(O_indices, O_values, Nx, Ny)
    iter = 20; % 20
    thrhld = 0.1; % 0.1
    diagcomp = 0.05; % 0.05

    % Create sparse matrix O from input indices and values
    M = sparse(O_indices(1, :), O_indices(2, :), O_values, Nx, Ny);
    disp('After creating sparse matrix:');


    O = M' * M;

    disp(size(O_indices));
    disp(size(O_values));
    disp(size(O));

    % Incomplete Cholesky factorization for preconditioning
    % M_pc = ichol(O, struct('diagcomp', 0.0001));  % further reducing the diagcomp will incur zero pivot error
    L = ichol(O, struct('diagcomp', diagcomp));
    disp('After incomplete Cholesky factorization:');

    % output
    [i, j, v] = find(L);
    O_inv_indices_i = i;
    O_inv_indices_j = j;
    O_inv_values = v;
end

