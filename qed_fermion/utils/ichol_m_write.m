function preconditioner(O_indices, O_values, Lx, Ly, Ltau)
    iter = 20; % 20
    thrhld = 0.1; % 0.1
    diagcomp = 0.05; % 0.05

    Nx = Lx * Ly * Ltau; % Total number of sites
    Ny = Nx; 

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

    filter = abs(O_inv_values) >= 1e-4;
    O_inv_indices_i = O_inv_indices_i(filter);
    O_inv_indices_j = O_inv_indices_j(filter);
    O_inv_values = O_inv_values(filter);

    % Prepare filename
    script_path = fileparts(mfilename('fullpath'));
    output_dir = fullfile(script_path, '..', 'preconditioners', 'pre_precon');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    filename = fullfile(output_dir, sprintf('L_mat_precon_%d_%d_%d.data', Lx, Ly, Ltau));

    % Combine data into a single matrix: each row is [i, j, value]
    data_to_write = [O_inv_indices_i, O_inv_indices_j, O_inv_values];

    % Write to file in ASCII format, space-separated, for easy reading in Python (e.g., with numpy.loadtxt)
    fid = fopen(filename, 'w');
    if fid == -1
        error('Cannot open file for writing: %s', filename);
    end
    % Write complex values: real and imaginary parts separately
    fprintf(fid, '%d %d %.7e %.7e\n', [O_inv_indices_i, O_inv_indices_j, real(O_inv_values), imag(O_inv_values)]');
    fclose(fid);
    disp(['Data written to ', filename]);
end

