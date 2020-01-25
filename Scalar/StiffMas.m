function K = StiffMas(elements, nodes, c, settings)
% STIFFMAS Create the global stiffness matrix K for a SCALAR problem in SERIAL computing.
%   STIFFMAS(elements,nodes,c) returns a sparse matrix K from finite element
%   analysis of scalar problems in a three-dimensional domain, where "elements"
%   is the connectivity matrix of size nelx8, "nodes" the nodal coordinates of
%   size Nx3, and "c" the material property for a linear isotropic material (scalar).
%
%   See also STIFFMASS, STIFFMAPS, SPARSE
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 05/12/2019. Version: 1.4. Name changed, Doc improved
% 	Modified: 21/01/2019. Version: 1.3
%   Created:  30/11/2018. Version: 1.0

iK = zeros(8,8,settings.nel,settings.dTE);      % Stores the rows' indices
jK = zeros(8,8,settings.nel,settings.dTE);      % Stores the columns' indices
Ke = zeros(8,8,settings.nel,settings.dTN);      % Stores the NNZ values
for e = 1:settings.nel                          % Loop over elements
    n = elements(e,:);                          % Nodes of the element 'e'
    X = nodes(n,:);                             % Nodal coordinates of the element 'e'
    ind = repmat(n,8,1);                        % Index for element 'e'
    iK(:,:,e) = ind';                           % Row index storage
    jK(:,:,e) = ind;                            % Columm index storage
    Ke(:,:,e) = Hex8scalars(X,c,settings.dTN);  % Element stiffness matrix compute & storage
end
K = AssemblyStiffMa(iK(:),jK(:),Ke(:),settings.dTE,settings.dTN); % Global stiffness matrix K assembly
