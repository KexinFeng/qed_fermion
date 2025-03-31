function [O_inv_indices, O_inv_values] = preconditioner(O_indices, O_values, Nx, Ny)
    iter = 20; % 20
    thrhld = 0.1; % 0.1
    diagcomp = 0.05; % 0.05

    % Create sparse matrix O from input indices and values
    M = sparse(O_indices(1, :), O_indices(2, :), O_values, Nx, Ny);

    O = M' * M;

    disp(size(O_indices));
    disp(size(O_values));
    disp(size(O));

    % Incomplete Cholesky factorization for preconditioning
    % M_pc = ichol(O, struct('diagcomp', 0.0001));  % further reducing the diagcomp will incur zero pivot error
    M_pc = ichol(O, struct('diagcomp', diagcomp));  % further reducing the diagcomp will incur zero pivot error
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

    % Neumann series approximation for matrix inverse
    for i = 1:iter
        M_temp = M_temp * M_itr;
        M_inv = M_inv + M_temp;
    end

    % ie. dd^-1 * L_inv
    M_inv = M_inv .* (1 ./ diag(dd));

    % Filter small elements to maintain sparsity directly on the sparse matrix
    O_inv1 = M_inv' * M_inv;
    disp('Sparsity of O_inv1:');
    disp(sprintf('%.2g, %d, %d', 1 - double(nnz(O_inv1)) / double(Nx * Ny), nnz(O_inv1), Nx * Ny));

    [i, j, v] = find(O_inv1);
    % v(abs(v) < 0.1) = 0; % Set small elements to zero
    % O_inv = sparse(i, j, v, Nx, Ny);
    % disp('Sparsity of O_inv:');
    % disp(sprintf('%.2g, %d, %d', 1 - double(length(v)) / double(Nx * Ny), length(v), Nx * Ny));

    % % Filter small elements to maintain sparsity
    % O_inv_ref = full(M_inv' * M_inv);
    % O_inv_ref = O_inv_ref - O_inv_ref .* ((abs(O_inv_ref)) < 0.1);
    % O_inv_ref = sparse(O_inv_ref);

    % % Assert that O_inv is equal to O_inv_ref
    % assert(isequal(O_inv, O_inv_ref), 'O_inv does not match O_inv_ref');

    % Gather data from GPU if computed on GPU
    % if gpuDeviceCount > 0
    %     O_inv = gather(O_inv);
    % end

    % Extract indices and values for COO sparse storage
    % [row, col, val] = find(O_inv);
    % O_inv_indices_ref = [row, col];
    % O_inv_values_ref = val;

    filter = abs(v) >= thrhld;
    O_inv_indices = gather([i(filter), j(filter)]);
    O_inv_values = gather(v(filter));
    disp('Sparsity of O_inv::');
    disp(sprintf('%.8g, %d, %d', 1 - length(O_inv_values)/double(Nx * Ny), length(O_inv_values), Nx * Ny));

    % % Assert that O_inv_indices and O_inv_values match their references
    % assert(isequal(O_inv_indices, O_inv_indices_ref), 'O_inv_indices does not match O_inv_indices_ref');
    % assert(isequal(O_inv_values, O_inv_values_ref), 'O_inv_values does not match O_inv_values_ref');
end