function K = StiffMaps(elements, nodes, c, settings)
% STIFFMAPS Create the global stiffness matrix tril(K) for a SCALAR problem in PARALLEL computing
% taking advantage of symmetry and GPU computing.
%   STIFFMAPS(elements,nodes,c,tbs) returns the lower-triangle of a sparse matrix
%   K from finite element analysis of scalar problems in a three-dimensional
%   domain taking advantage of symmetry and GPU computing, where "elements" is
%   the connectivity matrix of size 8xnel, "nodes" the nodal coordinates of size
%   3xN, "c" the material property for an isotropic material (scalar), and the
%   optional "tbs" refers to ThreadBlockSize (scalar).
%
%   See also STIFFMASS, STIFFMAS, SPARSE, ACCUMARRAY
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 07/12/2019. Version: 1.4. Variable number of inputs, Name changed, Doc improved
% 	Modified: 21/01/2019. Version: 1.3
%   Created:  13/12/2018. Version: 1.0

%% Inputs check
if ~(existsOnGPU(elements) && existsOnGPU(nodes))                               % Check if "elements" & "nodes" are on GPU memory
    error('Inputs "elements" and "nodes" must be on GPU memory. Use "gpuArray"');
elseif ( size(elements,1)~=8 || size(nodes,1)~=3 )                              % Check if "elements" & "nodes" are 8xnel & 3xN
    error('Input "elements" must be a 8xnel array, and "nodes" of size 3xN');
elseif ~( strcmp(settings.dTE,'uint32') || strcmp(settings.dTE,'uint64') || strcmp(settings.dTE,'double') )% Check data type for "elements"
    error('Input "elements" must be "uint32" or "uint64"');
elseif ~strcmp(settings.dTN,'double')                                                    % Check data type for "nodes"
    error('MATLAB only support "double" sparse matrix, i.e. "nodes" must be of type "double" ');
elseif ~isscalar(c)                                                             % Check input "c"
    error('Input "c" must be a SCALAR variable');
end

%% Index computation
[iK, jK] = IndexScalarsap(elements, settings);      % Row/column indices of tril(K)

%% Element matrix computation
Ke = Hex8scalarsap(elements, nodes, c, settings);	% Entries of tril(K)

%% Assembly of global sparse matrix on GPU
K = AssemblyStiffMa(iK, jK, Ke, settings.dTE, settings.dTN);	% Global stiffness matrix K assembly
