function K = AssemblyVectorSymGPU(elements,nodes,E,nu)
% Construction of the global stiffness matrix K (VECTOR-SYMMETRIC-DOUBLE-SPARSE)
N = size(nodes,1);                                  % Total number of nodes
[iK, jK] = IndexVectorGPU(elements);                % Row/column indices of tril(K)
Ke = Hex8vectorSymGPU(elements,nodes,E,nu);         % Entries of tril(K)
K = sparse(double(iK), double(jK), Ke, 3*N, 3*N);   % Assembly of global K
