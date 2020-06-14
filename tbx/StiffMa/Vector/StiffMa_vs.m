function K = StiffMa_vs(Mesh, MP, sets)
% STIFFMA_VS Create the global stiffness matrix K for a VECTOR (v) problem
% in SERIAL (s) computing.
%   STIFFMA_VS(Mesh, MP, sets) returns a sparse matrix K from finite element
%   analysis of vector problems in a three-dimensional domain, where Mesh
%   is a strucutre variable containing the field "Mesh.elements" which is
%   the connectivity matrix of size nelx8 and "Mesh.nodes" is the nodal
%   coordinates array of size Nx3. MP is also a struct with fields "MP.E"
%   (Young's modulus) and "MP.nu" (Poisson ratio) the   material property
%   for a linear isotropic  material. The struct "sets" must contain
%   several similation parameters:
%   - sets.dTE is the data precision of "Mesh.elements"
%   - sets.dTN is the data precision of "Mesh.nodes"
%   - sets.nel is the number of finite elements
%   - sets.edof is the number of DOFs per element.
%
%   See also STIFFMA_VSS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 07/02/2020. Version: 1.4. Name changed, Doc improved
% 	Modified: 28/01/2019. Version: 1.3
%   Created:  16/01/2019. Version: 1.0

%% Index computation
[iK, jK] = Index_va(Mesh.elements', sets);              % Row/column indices of tril(K)

%% Element stiffness matrix computation
Ke = eStiff_vsa(Mesh, MP, sets);                       % Entries of tril(K)

%% Assembly of global sparse matrix on CPU  % Lower Triangular sparse matrix
K = AssemblyStiffMa(iK, jK, Ke, sets);	% Global stiffness matrix K assembly

