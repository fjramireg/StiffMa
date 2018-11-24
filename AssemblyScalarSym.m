function K = AssemblyScalarSym(elements,nodes,c)
% Construction of the global stiffness matrix K (SCALAR-SYMMETRIC-DOUBLE)
N = size(nodes,1);                  % Total number of nodes
L = dNdrst;                         % Shape functions derivatives
nel = size(elements,1);             % Total number of elements
Ke = zeros(36,nel,'double');        % Stores the NNZ values
for e = 1:nel                       % Loop over elements
    n = elements(e,:);              % Nodes of the element 'e'
    X = nodes(n,:);                 % Nodal coordinates of the element 'e'
    Ke(:,e) = Hex8scalarSym(X,c,L); % Symmetric part of ke
end
[iK, jK] = IndexScalar(elements);   % Row/column indices of tril(K)
K = sparse(double(iK), double(jK), Ke(:)); % Assembly of global K
