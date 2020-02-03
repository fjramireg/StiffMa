function K = StiffMa_sss(Mesh, c, sets)
% STIFFMA_SSS Create the global stiffness matrix tril(K) for a SCALAR (s)
% problem in SERIAL (s) computing taking advantage of simmetry (s).
%   K=STIFFMA_SSS(Mesh,c,sets) returns the lower-triangle of a sparse matrix
%   K from finite element analysis of scalar problems in a three-dimensional
%   domain taking advantage of simmetry, where "Mesh.elements"  is the
%   connectivity matrix of size nelx8, "Mesh.nodes" the nodal coordinates
%   of size Nx3, and "c" (conductivity) is the material property for an
%   isotropic material. The struct "sets" must  contain several similation
%   parameters:
%   - sets.dTE is the data precision of "Mesh.elements"
%   - sets.dTN is the data precision of "Mesh.nodes"
%   - sets.nel is the number of finite elements
%   - sets.edof is the number of DOFs per element
%   - sets.sz  is the number of symmetry entries
%
%   See also STIFFMA_SS, STIFFMA_SPS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 07/12/2019. Version: 1.4. Name changed, Doc improved
% 	Modified: 21/01/2019. Version: 1.3
%   Created:  10/12/2018. Version: 1.0

%% Index computation - Row/column indices of tril(K)
[iK, jK] = Index_sss(Mesh.elements, sets);

%% Element stiffness matrix computation - Entries of tril(K)
Ke = eStiff_sssa(Mesh, c, sets);

%% Assembly of global sparse matrix on CPU - Global stiffness matrix K
K = AssemblyStiffMa(iK, jK, Ke, sets.dTE, sets.dTN);
