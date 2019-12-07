function K = StiffMaps(elements,nodes,c,tbs)
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

%% General declarations
dTE = classUnderlying(elements);            % "elements" data precision. Defines data type of [iK,jK]
dTN = classUnderlying(nodes);              	% "nodes" data precision. Defines data type of [Ke]

%% Inputs check
if ~(existsOnGPU(elements) && existsOnGPU(nodes)) % Check if "elements" & "nodes" are on GPU memory
    error('Inputs "elements" and "nodes" must be on GPU memory. Use "gpuArray"');
elseif ( size(elements,1)~=8 || size(nodes,1)~=3 )% Check if "elements" & "nodes" are 8xnel & 3xN
    error('Input "elements" must be a 8xnel array, and "nodes" of size 3xN');
elseif ~( strcmp(dTE,'int32') || strcmp(dTE,'uint32')... % Check data type for "elements"
        || strcmp(dTE,'int64')  || strcmp(dTE,'uint64') || strcmp(dTE,'double') )
    error('Input "elements" must be "int32", "uint32", "int64", "uint64" or "double" ');
elseif ~strcmp(dTN,'double')                      % Check data type for "nodes"
    error('MATLAB only support "double" sparse matrix, i.e. "nodes" must be of type "double" ');
elseif ~isscalar(c)                               % Check input "c"
    error('Input "c" must be a SCALAR variable');
end

%% Index computation
[iK, jK] = IndexScalarsap(elements, tbs);   % Row/column indices of tril(K)

%% Element matrix computation
Ke = Hex8scalarsap(elements,nodes,c, tbs);  % Entries of tril(K)

%% Assembly of global sparse matrix on GPU
K = accumarray([iK,jK], Ke, [], [], [], 1); % Stiffness matrix K
