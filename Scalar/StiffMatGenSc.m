function K = StiffMatGenSc(elements,nodes,c)
% STIFFMATGENSC Create the global stiffness matrix for a SCALAR problem.
%   STIFFMATGENSC(elements,nodes,c) returns a sparse matrix K from finite
%   element analysis of scalar problems in a three-dimensional domain,
%   where "elements" is the connectivity matrix, "nodes" the nodal
%   coordinates, and "c" the material property for an isotropic material.
%
%   See also SPARSE, STIFFMATGENSCSYMCPU, STIFFMATGENSCSYMCPUP
%
%   For more information, see <a href="matlab:
%   web('https://github.com/fjramireg/MatGen')">the MatGen Web site</a>.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created: 30/11/2018. Modified: 21/01/2019. Version: 1.3

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
