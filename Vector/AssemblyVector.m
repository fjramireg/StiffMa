%  * ====================================================================*/
% ** This function was developed by:
%  *          Francisco Javier Ramirez-Gil
%  *          Universidad Nacional de Colombia - Medellin
%  *          Department of Mechanical Engineering
%  *
%  ** Please cite this code as:
%  *
%  ** Date & version
%  *      Created: 16/01/2018. Last modified: 21/01/2019
%  *      V 1.3
%  *
%  * ====================================================================*/

function K = AssemblyVector(elements,nodes,E,nu)
% Construction of the global stiffness matrix K for a VECTOR problem
N = size(nodes,1);                  % Total number of nodes
nel = size(elements,1);             % Total number of elements
D = MaterialMatrix(E,nu,'double');  % Material matrix (isotropic)
iK = zeros(24,24,nel,'double');     % Stores the rows' indices
jK = zeros(24,24,nel,'double');     % Stores the columns' indices
Ke = zeros(24,24,nel,'double');     % Stores the NNZ values
for e = 1:nel                       % Loop over elements
    n = elements(e,:);              % Nodes of the element 'e'
    X = nodes(n,:);                 % Nodal coordinates of the element 'e'
    dof = [3*n-2; 3*n-1; 3*n];      % DOFs of the element 'e'
    ind = repmat(dof(:),1,24);      % Index for element 'e'
    iK(:,:,e) = ind;                % Row index storage
    jK(:,:,e) = ind';               % Columm index storage
    Ke(:,:,e) = Hex8vector(X,D);    % Element stiffnes matrix storage
end
K = sparse(iK(:),jK(:),Ke(:),3*N,3*N); % Assembly of the global stiffness matrix
