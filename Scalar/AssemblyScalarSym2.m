%  * ====================================================================*/
% ** This function was developed by:
%  *          Francisco Javier Ramirez-Gil
%  *          Universidad Nacional de Colombia - Medellin
%  *          Department of Mechanical Engineering
%  *
%  ** Please cite this code as:
%  *
%  ** Date & version
%  *      30/11/2018.
%  *      V 1.2
%  *
%  * ====================================================================*/

function K = AssemblyScalarSym2(elements,nodes,c)
% Construction of the global stiffness matrix K (SCALAR-SYMMETRIC-DOUBLE-SPARSE-ACCUMARRAY)
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
K = accumarray([iK,jK], Ke(:), [N,N], [], [], 1); % Assembly of global K
