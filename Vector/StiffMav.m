function K = StiffMav(elements,nodes,E,nu)
% STIFFMAV Create the global stiffness matrix K for a VECTOR problem in SERIAL computing.
%   STIFFMAV(elements,nodes,E,nu) returns a sparse matrix K from finite element
%   analysis of vector problems in a three-dimensional domain, where "elements"
%   is the connectivity matrix of size nelx8, "nodes" the nodal coordinates
%   array of size Nx3, and "E" (Young's modulus) and "nu" (Poisson ratio) the
%   material property for a linear isotropic material.
%
%   See also SPARSE, STIFFMATGENVCSYMCPU, STIFFMATGENVCSYMCPUP
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 18/12/2019. Version: 1.4. Name changed, Doc improved
% 	Modified: 28/01/2019. Version: 1.3
%   Created:  16/01/2019. Version: 1.0

dTypeInd = class(elements);         % Data type (precision) for index computation
dTypeKe = class(nodes);             % Data type (precision) for ke computation
N = size(nodes,1);                  % Total number of nodes
nel = size(elements,1);             % Total number of elements
D = MaterialMatrix(E,nu,dTypeKe);   % Material matrix (isotropic)
iK = zeros(24,24,nel,dTypeInd);     % Stores the rows' indices
jK = zeros(24,24,nel,dTypeInd);     % Stores the columns' indices
Ke = zeros(24,24,nel,dTypeKe);      % Stores the NNZ values
for e = 1:nel                       % Loop over elements
    n = elements(e,:);              % Nodes of the element 'e'
    X = nodes(n,:);                 % Nodal coordinates of the element 'e'
    dof = [3*n-2; 3*n-1; 3*n];      % DOFs of the element 'e'
    ind = repmat(dof(:),1,24);      % Index for element 'e'
    iK(:,:,e) = ind;                % Row index storage
    jK(:,:,e) = ind';               % Columm index storage
    Ke(:,:,e) = Hex8vectors(X,D);   % Element stiffnes matrix compute & storage
end
K = sparse(iK(:),jK(:),Ke(:),3*N,3*N); % Assembly of the global stiffness matrix
