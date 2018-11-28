function K = AssemblyScalarSymGPU2(elements,nodes,c)
% Construction of the global stiffness matrix K (SCALAR-SYMMETRIC-DOUBLE-ACCUMARRAY)
N = size(nodes,1);                              % Total number of nodes
[iK, jK] = IndexScalar(elements);               % Row/column indices of tril(K)
Ke = Hex8scalarSymGPU(elements,nodes,c);        % Entries of tril(K)
K = sparse(double(iK), double(jK), Ke, N, N);   % Assembly of global K
