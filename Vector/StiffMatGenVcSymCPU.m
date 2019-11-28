function K = StiffMatGenVcSymCPU(elements,nodes,E,nu)
% STIFFMATGENVCSYMCPU Create the global stiffness matrix for a VECTOR
% problem taking advantage of simmetry.
%   STIFFMATGENVCSYMCPU(elements,nodes,E,nu) returns the lower-triangle of
%   a sparse matrix K from finite element analysis of vector problems in a
%   three-dimensional domain taking advantage of simmetry, where "elements"
%   is the connectivity matrix, "nodes" the nodal coordinates, and "E"
%   (Young's modulus) and "nu" (Poisson ratio) the material property for an
%   isotropic material.
%
%   See also SPARSE, ACCUMARRAY, STIFFMATGENVC, STIFFMATGENVCSYMCPUP
%
%   For more information, see <a href="matlab:
%   web('https://github.com/fjramireg/MatGen')">the MatGen Web site</a>.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created: 16/01/2019. Modified: 28/01/2019. Version: 1.3

%% General declarations
N = size(nodes,1);                          % Total number of nodes (DOFs)
dTE = class(elements);                      % "elements" data precision
dTN = class(nodes);                         % "nodes" data precision

%% Index computation
[iK, jK] = IndexVectorSymCPU(elements);     % Row/column indices of tril(K)

%% Element stiffness matrix computation
Ke = Hex8vectorSymCPU(elements,nodes,E,nu);	% Entries of tril(K)

%% Assembly of global sparse matrix on CPU
K = AssemblyStiffMat(iK,jK,Ke,3*N,dTE,dTN); % Triangular sparse matrix



