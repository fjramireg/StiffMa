function K = StiffMatGenScSymGPU(elements,nodes,c)
% STIFFMATGENSCSYMGPU Create the global stiffness matrix for a SCALAR
% problem taking advantage of simmetry and GPU computing.
%   STIFFMATGENSCSYMGPU(elements,nodes,c) returns the lower-triangle of a
%   sparse matrix K from finite element analysis of scalar problems in a
%   three-dimensional domain taking advantage of simmetry and GPU computing,
%   where "elements" is the connectivity matrix, "nodes" the nodal coordinates,
%   and "c" the material property for an isotropic material.
%
%   See also SPARSE, ACCUMARRAY, STIFFMATGENSC, STIFFMATGENSCSYMCPU, STIFFMATGENSCSYMCPUP
%
%   For more information, see <a href="matlab:
%   web('https://github.com/fjramireg/MatGen')">the MatGen Web site</a>.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created: 13/12/2018. Modified: 21/01/2019. Version: 1.3

%% General declarations
dTE = classUnderlying(elements);            % "elements" data precision. Defines data type of [iK,jK]
dTN = classUnderlying(nodes);              	% "nodes" data precision. Defines data type of [Ke]
N = size(nodes,2);                          % Total number of nodes (DOFs)

%% Inputs check
if ~(existsOnGPU(elements) && existsOnGPU(nodes)) % Check if "elements" & "nodes" are on GPU memory
    error('Inputs "elements" and "nodes" must be on GPU memory. Use "gpuArray"');
elseif ( size(elements,1)~=8 || size(nodes,1)~=3 )% Check if "elements" & "nodes" are 8xnel & 3xnnod.
    error('Input "elements" must be a 8xnel array, and "nodes" of size 3xnnod');
elseif ~( strcmp(dTE,'int32') || strcmp(dTE,'uint32')... % Check data type for "elements"
        || strcmp(dTE,'int64')  || strcmp(dTE,'uint64') || strcmp(dTE,'double') )
    error('Error. Input "elements" must be "int32", "uint32", "int64", "uint64" or "double" ');
elseif ~strcmp(dTN,'double')                      % Check data type for "nodes"
    error('MATLAB only support "double" sparse matrix, i.e. "nodes" must be of type "double" ');
elseif ~isscalar(c)                               % Check input "c"
    error('Error. Input "c" must be a SCALAR variable');
end

%% Index computation
[iK, jK] = IndexScalarSymGPU(elements);     % Row/column indices of tril(K)

%% Element matrix computation
Ke = Hex8scalarSymGPU(elements,nodes,c);   	% Entries of tril(K)

%% Assembly of global sparse matrix on GPU
K = AssemblyStiffMat(iK,jK,Ke,N,dTE,dTN);
