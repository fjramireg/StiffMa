function [iK, jK] = Index_spsa(elements, sets)
% INDEX_SPSA Computes the row/column indices of tril(K) for a SCALAR (s) problem
% in PARALLEL (p) GPU computing taking advantage of symmetry (s) to return ALL (a)
% indices for the mesh.
%   [iK, jK]=INDEX_SPSA(elements, sets) returns the rows "iK" and columns "jK"
%   position of all element stiffness matrices in the global system for a finite
%   element analysis of a scalar problem in a three-dimensional domain taking
%   advantage of symmetry and GPU computing, where "elements" is the
%   connectivity matrix of size 8xnel and dType is the data type defined to the
%   "elements" array.  The struct "sets" must contain several similation
%   parameters: 
%   - sets.dTE is the data precision of "Mesh.elements"
%   - sets.dTN is the data precision of "Mesh.nodes"
%   - sets.nel is the number of finite elements
%   - sets.sz  is the number of symmetry entries
%   - sets.tbs is the Thread Block Size
%   - sets.numSMs is the number of multiprocessors on the device
%   - sets.WarpSize is the warp size
%
%   See also STIFFMA_SPS, INDEX_SSS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 08/02/2020. Version: 1.4. Error fix to use 'uint64'
% 	Modified: 28/01/2020. Version: 1.3. Name changed, Doc improved
% 	Modified: 21/01/2019. Version: 1.2
%   Created:  30/11/2018. Version: 1.0

% MATLAB KERNEL CREATION
if strcmp(sets.dTE,'uint32')               % uint32
    ker = parallel.gpu.CUDAKernel('Index_sps.ptx',...                           % PTXFILE
        'const unsigned int*,const unsigned int,unsigned int*,unsigned int*',...% C prototype for kernel
        'IndexScalarGPUIj');                                                    % Specify entry point
elseif strcmp(sets.dTE,'uint64')           % uint64
    ker = parallel.gpu.CUDAKernel('Index_sps.ptx',...
        'const unsigned long long int *, const unsigned long long int, unsigned long long int *, unsigned long long int *',...
        'IndexScalarGPUIy');
else
    error('Not supported data type. Use only one of this: uint32 & uint64');
end

% MATLAB KERNEL CONFIGURATION
if (sets.tbs > ker.MaxThreadsPerBlock || mod(sets.tbs, sets.WarpSize) )
    sets.tbs = ker.MaxThreadsPerBlock;
    if  mod(sets.tbs, sets.WarpSize)
        sets.tbs = sets.tbs - mod(sets.tbs, sets.WarpSize);
    end
end
ker.ThreadBlockSize = [sets.tbs, 1, 1];             % Threads per block
ker.GridSize = [sets.WarpSize*sets.numSMs, 1, 1];  	% Blocks per grid

% INITIALIZATION OF GPU VARIABLES
iK  = zeros(36*sets.nel, 1, sets.dTE, 'gpuArray');	% Stores row indices (initialized directly on GPU)
jK  = zeros(36*sets.nel, 1, sets.dTE, 'gpuArray');	% Stores column indices (initialized directly on GPU)

% MATLAB KERNEL CALL
[iK, jK] = feval(ker, elements, sets.nel, iK, jK);  % GPU code execution
