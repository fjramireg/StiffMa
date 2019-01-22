function K = AssemblyScalarSym(elements,nodes,c)
% ASSEMBLYSCALARSYM Create the global stiffness matrix for a SCALAR
% problem taking advantage of simmetry.
%   ASSEMBLYSCALARSYM(elements,nodes,c) returns the lower-triangle of a
%   sparse matrix K from finite element analysis of scalar problems in a
%   three-dimensional domain taking advantage of simmetry, where "elements"
%   is the connectivity matrix, "nodes" the nodal coordinates, and "c" the
%   material property for an isotropic material.
%
%   See also SPARSE, ACCUMARRAY, ASSEMBLYSCALAR, ASSEMBLYSCALARSYMGPU
%
%   For more information, see <a href="matlab:
%   web('https://github.com/fjramireg/MatGen')">the MatGen Web site</a>.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created: 10/12/2018. Modified: 21/01/2019. Version: 1.3

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
