%  * ====================================================================*/
% ** This function was developed by:
%  *          Francisco Javier Ramirez-Gil
%  *          Universidad Nacional de Colombia - Medellin
%  *          Department of Mechanical Engineering
%  *
%  ** Please cite this code as:
%  *
%  ** Date & version
%  *      Created: 10/12/2018. Last modified: 21/01/2019
%  *      V 1.3
%  *
%  * ====================================================================*/

function K = AssemblyScalarSym(elements,nodes,c)
% Construction of the global stiffness matrix K (SCALAR-SYMMETRIC)
N = size(nodes,1);                  % Total number of nodes (DOFs)
L = dNdrst(class(nodes));           % Shape functions derivatives
nel = size(elements,1);             % Total number of elements
Ke = zeros(36, nel, class(nodes));  % Stores the NNZ values
for e = 1:nel                       % Loop over elements
    n = elements(e,:);              % Nodes of the element 'e'
    X = nodes(n,:);                 % Nodal coordinates of the element 'e'
    Ke(:,e) = Hex8scalarSym(X,c,L); % Symmetric part of ke
end
[iK, jK] = IndexScalarSym(elements);% Row/column indices of tril(K)

% Assembly of global sparse matrix K
if ( isa(elements,'double') && isa(nodes,'double') )
    K = sparse(iK, jK, Ke(:), N, N);
else
    K = accumarray([iK,jK], Ke(:), [N,N], [], [], 1);
end
