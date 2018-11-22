function K = AssemblyVector(elements,nodes,E,nu)
% Assembly of the global stiffnes matrix K for a SCALAR problem
N = size(nodes,1);              % Total number of nodes
nel = size(elements,1);         % Total number of elements
D = MaterialMatrix(E,nu);       % Material matrix
iK = zeros(24,24,nel,'double'); % Stores the rows' indices
jK = zeros(24,24,nel,'double'); % Stores the columns' indices
Ke = zeros(24,24,nel,'double'); % Stores the NNZ values
for e = 1:nel                   % Loop over elements
    n = elements(e,:);          % Nodes of the element e
    X = nodes(n,:);             % Nodal coordinates of the element e
    dof = [3*n-2; 3*n-1; 3*n];  % DOFs of the element e
    ind = repmat(dof(:),1,24);  % Index for element e
    iK(:,:,e) = ind;            % Row index
    jK(:,:,e) = ind';           % Columm index
    Ke(:,:,e) = Hex8vector(X,D); % Element stiffnes matrix    
end
K = sparse(iK(:),jK(:),Ke(:),3*N,3*N);  % Assembly process
