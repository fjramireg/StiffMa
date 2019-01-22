function K = AssemblyVectorSym(elements,nodes,E,nu)
% ASSEMBLYVECTORSYM Create the global stiffness matrix for a VECTOR problem
% taking advantage of simmetry.
%   ASSEMBLYVECTORSYM(elements,nodes,E,nu) returns the lower-triangle of a
%   sparse matrix K from finite element analysis of vector problems in a
%   three-dimensional domain taking advantage of simmetry, where "elements"
%   is the connectivity matrix, "nodes" the nodal coordinates, and "E"
%   (Young's modulus) and "nu" (Poisson ratio) the material property for an
%   isotropic material.
%
%   See also SPARSE, ACCUMARRAY, ASSEMBLYVECTOR, ASSEMBLYVECTORSYMGPU
%
%   For more information, see <a href="matlab:
%   web('https://github.com/fjramireg/MatGen')">the MatGen Web site</a>.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created: 16/01/2019. Modified: 21/01/2019. Version: 1.3

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
