function Ke = eStiff_vsa(Mesh, MP, sets)
% ESTIFF_VSA Computes the element stiffness matrices for a VECTOR (v) problem
% in SERIAL (s) computing on the CPU and returning ALL (a) ke for the mesh.
%   Ke = ESTIFF_VSA(Mesh, MP, sets) returns a full matrix Ke from finite element
%   analysis of scalar problems in a three-dimensional domain containing all
%   element stiffnes matrices, where Mesh is a strucutre variable containing the
%   field "Mesh.elements" which is the connectivity matrix of size nelx8 and
%   "Mesh.nodes" is the nodal coordinates array of size Nx3. MP is also a struct
%   with fields "MP.E" (Young's modulus) and "MP.nu" (Poisson ratio) the
%   material property for a linear isotropic  material. The struct "sets" must
%   contain several similation parameters:
%   - sets.dTE is the data precision of "Mesh.elements"
%   - sets.dTN is the data precision of "Mesh.nodes"
%   - sets.nel is the number of finite elements
%   - sets.edof is the number of DOFs per element.
%
%   See also STIFFMA_VS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 07/02/2020. Version: 1.4. Name changed, Doc improved
% 	Modified: 28/01/2019. Version: 1.3
%   Created:  16/01/2019. Version: 1.0

D = DMatrix(MP.E, MP.nu, sets.dTN);                                     % Material matrix (isotropic)
Ke = zeros(sets.edof, sets.edof, sets.nel, sets.dTN);                   % Stores the NNZ values
for e = 1:sets.nel                                                      % Loop over elements
    Ke(:,:,e) = eStiff_vs(Mesh.nodes(Mesh.elements(e,:),:),D,sets.dTN);	% Element stiffnes matrix compute & storage
end
Ke = Ke(:);
