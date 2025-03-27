function [O_inv_indices, O_inv_values] = preconditioner(O_indices, O_values, Nx, Ny)
    % Create sparse matrix O from input indices and values
    O = sparse(O_indices(1, :), O_indices(2, :), O_values, Nx, Ny);

    % Incomplete Cholesky factorization for preconditioning
    M_pc = ichol(O, struct('diagcomp', 0.05));
    dd = diag(diag(M_pc));
    
    % Check if a GPU is available
    if gpuDeviceCount > 0
        I = gpuArray(speye(Nx)); % Use GPU if available
        M_itr = gpuArray(sparse(I - M_pc / dd));
        M_temp = gpuArray(I);
        M_inv = gpuArray(I);
    else
        I = speye(Nx); % Use CPU otherwise
        M_itr = sparse(I - M_pc / dd);
        M_temp = I;
        M_inv = I;
    end

    for i = 1:20
        M_temp = M_temp * M_itr;
        M_inv = M_inv + M_temp;
    end

    M_inv = M_inv .* (1 ./ diag(dd));
    O_inv = full(M_inv' * M_inv);

    % Filter small elements to maintain sparsity
    O_inv = O_inv - O_inv .* ((abs(O_inv)) < 0.1);
    O_inv = sparse(O_inv);

    % Gather data from GPU if computed on GPU
    if gpuDeviceCount > 0
        O_inv = gather(O_inv);
    end

    % Extract indices and values for COO sparse storage
    [row, col, val] = find(O_inv);
    O_inv_indices = [row, col];
    O_inv_values = val;
end