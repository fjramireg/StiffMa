%  * ====================================================================*/
% ** This function was developed by:
%  *          Francisco Javier Ramirez-Gil
%  *          Universidad Nacional de Colombia - Medellin
%  *          Department of Mechanical Engineering
%  *
%  ** Please cite this code as:
%  *
%  ** Date & version
%  *      16/01/2019.
%  *      V 1.2
%  *
%  * ====================================================================*/

function K = AssemblyVectorSym(elements,nodes,E,nu)
% Construction of the global stiffness matrix K (VECTOR-SYMMETRIC)
dType = class(nodes);                   % Data type of "nodes"
N = size(nodes,1);                      % Total number of nodes
L = dNdrst(dType);                      % Shape functions derivatives
nel = size(elements,1);                 % Total number of elements
D = MaterialMatrix(E,nu,dType);         % Material matrix (isotropic)
Ke = zeros(300,nel,dType);              % Stores the NNZ values
for e = 1:nel                           % Loop over elements
    n = elements(e,:);                  % Nodes of the element 'e'
    X = nodes(n,:);                     % Nodal coordinates of the element 'e'
    Ke(:,e) = Hex8vectorSym(X,D,L);     % Element stiffnes matrix storage
end
[iK, jK] = IndexVectorSym(elements);    % Row/column indices of tril(K)

% Assembly of global sparse matrix K
if ( isa(elements,'double') && isa(nodes,'double') )
    K = sparse(iK, jK, Ke(:), 3*N, 3*N);
else
    K = accumarray([iK,jK], Ke(:), [3*N,3*N], [], [], 1);
end
