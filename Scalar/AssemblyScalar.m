function K = AssemblyScalar(elements,nodes,c)
% Construction of the global stiffness matrix K for a SCALAR problem
N = size(nodes,1);                  % Total number of nodes (DOFs)
nel = size(elements,1);             % Total number of elements
iK = zeros(8,8,nel,'double');       % Stores the rows' indices
jK = zeros(8,8,nel,'double');       % Stores the columns' indices
Ke = zeros(8,8,nel,'double');       % Stores the NNZ values
for e = 1:nel                       % Loop over elements
    n = elements(e,:);              % Nodes of the element 'e'
    X = nodes(n,:);                 % Nodal coordinates of the element 'e'
    ind = repmat(n,8,1);            % Index for element 'e'
    iK(:,:,e) = ind';               % Row index storage
    jK(:,:,e) = ind;                % Columm index storage
    Ke(:,:,e) = Hex8scalar(X,c);    % Element stiffness matrix storage
end
K = sparse(iK(:),jK(:),Ke(:),N,N);  % Assembly of the global stiffness matrix
