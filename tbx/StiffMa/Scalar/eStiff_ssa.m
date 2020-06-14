function Ke = eStiff_ssa(Mesh, c, sets)
% ESTIFF_SSA Computes the element stiffness matrices for a SCALAR (s) problem
% in SERIAL (s) computing on the CPU and returning ALL (a) ke for the mesh.
%   Ke = ESTIFF_SSA(Mesh, c, sets) returns a full matrix Ke from finite
%   element analysis of scalar problems in a three-dimensional domain containing
%   all element stiffnes matrices, where "Mesh.elements" is the connectivity
%   matrix of size nelx8, "Mesh.nodes" the nodal coordinates of size Nx3, and
%   "c" (conductivity) is the material property for a linear isotropic material.
%   The struct "sets" must  contain several similation parameters:
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
% 	Modified: 07/02/2020. Version: 1.4. Name changed, Doc improved
% 	Modified: 21/01/2019. Version: 1.3
%   Created:  30/11/2018. Version: 1.0

Ke = zeros(sets.edof, sets.edof, sets.nel, sets.dTN);                   % Stores the NNZ values
for e = 1:sets.nel                                                      % Loop over elements
    Ke(:,:,e) = eStiff_ss(Mesh.nodes(Mesh.elements(e,:),:),c,sets.dTN); % Element stiffness matrix compute & storage
end
Ke = Ke(:);
