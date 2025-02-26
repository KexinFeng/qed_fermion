A = [5 -2 0 -2 -2;
     -2 5 -2 0 0;
     0 -2 5 -2 0;
     -2 0 -2 5 -2;
     -2 0 0 -2 5];
L = chol(A)

L'*L - A


disp('--- sparse ----')
A = sparse(A);

R = ichol(A);
full(R)

R * R' - A

full(R * R')
full((R*R').*spones(A))

(R*R').*spones(A) - A


disp('-----')
N = 100;
A = delsq(numgrid('S',N));

L = ichol(A);
norm(A-L*L','fro')./norm(A,'fro')

norm(A-(L*L').*spones(A),'fro')./norm(A,'fro')

dbstop = 1
