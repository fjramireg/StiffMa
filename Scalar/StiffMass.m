function K = StiffMass(elements,nodes,c)
% STIFFMASS Create the global stiffness matrix tril(K) for a SCALAR problem in SERIAL computing
% taking advantage of simmetry.
%   STIFFMASS(elements,nodes,c) returns the lower-triangle of a sparse matrix K
%   from finite element analysis of scalar problems in a three-dimensional
%   domain taking advantage of simmetry, where "elements" is the connectivity
%   matrix of size nelx8, "nodes" the nodal coordinates of size Nx3, and "c" the
%   material property for an isotropic material (scalar).
%
%   See also STIFFMAS, STIFFMAPS, SPARSE, ACCUMARRAY
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 07/12/2019. Version: 1.4. Name changed, Doc improved
% 	Modified: 21/01/2019. Version: 1.3
%   Created:  10/12/2018. Version: 1.0


%% Data type (precision)
dTE = class(elements);                      % for index computation
dTN = class(nodes);                         % for ke computation

%% Index computation
[iK, jK] = IndexScalarsas(elements,dTE);    % Row/column indices of tril(K)

%% Element stiffness matrix computation
Ke = Hex8scalarsas(elements,nodes,c,dTN);   % Entries of tril(K)

%% Assembly of global sparse matrix on CPU
K = AssemblyStiffMa(iK,jK,Ke,dTE,dTN);      % Global stiffness matrix K assembly
