function K = AssemblyVectorSymGPU(elements,nodes,E,nu)
% Construction of the global stiffness matrix K (VECTOR-SYMMETRIC-DOUBLE-SPARSE)
N = size(nodes,1);                      % Total number of nodes
L = dNdrst;                             % Shape functions derivatives
nel = size(elements,1);                 % Total number of elements
D = MaterialMatrix(E,nu);               % Material matrix (isotropic)
Ke = zeros(300,nel,'double');           % Stores the NNZ values
for e = 1:nel                           % Loop over elements
    n = elements(e,:);                  % Nodes of the element 'e'
    X = nodes(n,:);                     % Nodal coordinates of the element 'e'
    Ke(:,e) = Hex8vectorSym(X,D,L);     % Element stiffnes matrix storage
end
[iK, jK] = IndexVector(elements);       % Row/column indices of tril(K)
KE = Hex8vectorSymGPU(elements,nodes,E,nu); 
K = sparse(iK, jK, Ke(:), 3*N, 3*N);    % Assembly of global K
