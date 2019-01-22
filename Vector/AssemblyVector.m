function K = AssemblyVector(elements,nodes,E,nu)
% ASSEMBLYVECTOR Create the global stiffness matrix for a VECTOR problem.
%   ASSEMBLYVECTOR(elements,nodes,E,nu) returns a sparse matrix K from
%   finite element analysis of vector problems in a three-dimensional
%   domain, where "elements" is the connectivity matrix, "nodes" the nodal
%   coordinates, and "E" (Young's modulus) and "nu" (Poisson ratio) the
%   material property for an isotropic material. 
%
%   See also SPARSE, ASSEMBLYVECTORSYM, ASSEMBLYVECTORSYMGPU
%
%   For more information, see <a href="matlab:
%   web('https://github.com/fjramireg/MatGen')">the MatGen Web site</a>.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created: 16/01/2019. Modified: 21/01/2019. Version: 1.3

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
