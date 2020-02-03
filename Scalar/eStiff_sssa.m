function Ke = eStiff_sssa(Mesh, c, sets)
% ESTIFF_SSSA Computes the element stiffness matrices for a SCALAR (s)
% problem in SERIAL (s) computing on the CPU taking advantage of symmetry
% (s) and returning ALL (a) ke for the mesh.
%   Ke=ESTIFF_SSSA(Mesh,c,sets) returns the element stiffness matrix "ke"
%   for all elements in a finite element analysis of scalar problems in a
%   three-dimensional domain taking advantage of symmetry but in a serial
%   manner on the CPU,  where "Mesh.elements" is the connectivity matrix of
%   size nelx8, "Mesh.nodes" the nodal coordinates of size Nx3, and "c" the
%   material property for an isotropic material (scalar). The struct "sets"
%   must contain several similation parameters:
%   - sets.dTN is the data precision of "Mesh.nodes"
%   - sets.nel is the number of finite elements
%   - sets.sz  is the number of symmetry entries
%
%   See also STIFFMA_SSS ESTIFF_SSS, ESTIFF_SPSA
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 07/12/2019. Version: 1.4. Name changed, Doc improved
% 	Modified: 22/01/2019. Version: 1.3
%   Created:  30/11/2018. Version: 1.0

L = dNdrst(sets.dTN);                       % Shape functions derivatives
Ke = zeros(sets.sz, sets.nel, sets.dTN);	% Stores the NNZ values
for e = 1:sets.nel                          % Loop over elements
    n = Mesh.elements(e,:);               	% Nodes of the element 'e'
    X = Mesh.nodes(n,:);                   	% Nodal coordinates of the element 'e'
    Ke(:,e) = eStiff_sss(X,c,L,sets.dTN);   % Symmetric part of ke
end
Ke = Ke(:);
