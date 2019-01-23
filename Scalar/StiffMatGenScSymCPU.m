function K = StiffMatGenScSymCPU(elements,nodes,c)
% STIFFMATGENSCSYMCPU Create the global stiffness matrix for a SCALAR
% problem taking advantage of simmetry.
%   STIFFMATGENSCSYMCPU(elements,nodes,c) returns the lower-triangle of a
%   sparse matrix K from finite element analysis of scalar problems in a
%   three-dimensional domain taking advantage of simmetry, where "elements"
%   is the connectivity matrix, "nodes" the nodal coordinates, and "c" the
%   material property for an isotropic material.
%
%   See also SPARSE, ACCUMARRAY, STIFFMATGENSC, STIFFMATGENSCSYMGPU
%
%   For more information, see <a href="matlab:
%   web('https://github.com/fjramireg/MatGen')">the MatGen Web site</a>.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created: 10/12/2018. Modified: 21/01/2019. Version: 1.3

%% General declarations
N = size(nodes,1);                      % Total number of nodes (DOFs)
dTE = class(elements);                  % "elements" data precision
dTN = class(nodes);                     % "nodes" data precision

%% Index computation
[iK, jK] = IndexScalarSymCPU(elements);    % Row/column indices of tril(K)

%% Element stiffness matrix computation
Ke = Hex8scalarSymCPU(elements,nodes,c);% Entries of tril(K)

%% Assembly of global sparse matrix on CPU
K = AssemblyStiffMat(iK,jK,Ke(:),N,dTE,dTN);
