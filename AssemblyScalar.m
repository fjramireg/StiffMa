function K = AssemblyScalar(elements,nodes,c)
% Global stiffnes matrix K for a SCALAR problem
N = size(nodes,1);      % Total number of nodes
nel = size(elements,1); % Total number of elements
iK = zeros(64,64,nel);  % Stores the rows' indices
jK = zeros(64,64,nel);  % Stores the columns' indices
Ke = zeros(64,64,nel);  % Stores the NNZ values
% Assembly process
for e = 1:nel        
    n = elements(e,:);  % Nodes of the element e    
    X = nodes(n,:);     % Nodal coordinates of the element e    
    i_ind = repmat(n,8,1);
    iK(:,:,e) = i_ind;
    jK(:,:,e) = i_ind';
    KE(:,:,e) = Hex8scalar(X,c);
end
K = sparse(iK,jK,Ke,N,N);
