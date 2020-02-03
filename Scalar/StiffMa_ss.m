function K = StiffMa_ss(Mesh, c, sets)
% STIFFMA_SS Create the global stiffness matrix K for a SCALAR (s) problem
% in SERIAL (s) computing.
%   K=STIFFMAS(Mesh, c, sets) returns a sparse matrix K from finite element
%   analysis of scalar problems in a three-dimensional domain, where
%   "Mesh.elements" is the connectivity matrix of size nelx8, "Mesh.nodes"
%   the nodal coordinates of size Nx3, and "c" (conductivity) is the
%   material property for a linear isotropic material. The struct "sets"
%   must  contain several similation parameters:
%   - sets.dTE is the data precision of "Mesh.elements"
%   - sets.dTN is the data precision of "Mesh.nodes"
%   - sets.nel is the number of finite elements
%   - sets.edof is the number of DOFs per element
%   - sets.sz  is the number of symmetry entries
%
%   See also STIFFMA_SSS, STIFFMA_SPS, ESTIFF_SS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 05/12/2019. Version: 1.4. Name changed, Doc improved
% 	Modified: 21/01/2019. Version: 1.3
%   Created:  30/11/2018. Version: 1.0

iK = zeros(sets.edof, sets.edof, sets.nel, sets.dTE);       % Stores the rows' indices
jK = zeros(sets.edof, sets.edof, sets.nel, sets.dTE);       % Stores the columns' indices
Ke = zeros(sets.edof, sets.edof, sets.nel, sets.dTN);       % Stores the NNZ values
for e = 1:sets.nel                                          % Loop over elements
    n = Mesh.elements(e,:);                                 % Nodes of the element 'e'
    X = Mesh.nodes(n,:);                                    % Nodal coordinates of the element 'e'
    ind = repmat(n,sets.edof,1);                            % Index for element 'e'
    iK(:,:,e) = ind';                                       % Row index storage
    jK(:,:,e) = ind;                                        % Columm index storage
    Ke(:,:,e) = eStiff_ss(X,c,sets.dTN);                    % Element stiffness matrix compute & storage
end
K = AssemblyStiffMa(iK(:),jK(:),Ke(:),sets.dTE,sets.dTN);	% Global stiffness matrix K assembly
