function [iK, jK] = Index_vps(elements, sets)
% INDEX_VPS Compute the row/column indices of tril(K) in a vector (v)
% problem using parallel (p) GPU computing using symmety (s).
%   INDEX_VPS(elements, sets) returns the rows "iK" and columns "jK"
%   position of all element stiffness matrices in the global system for a
%   finite element analysis of a vector problem in a three-dimensional
%   domain taking advantage of symmetry, where "elements" is the
%   connectivity matrix of size 8xnel. The struct "sets" must contain
%   several similation parameters: 
%   - sets.dTE is the data precision of "elements"
%   - sets.nel is the number of finite elements
%   - sets.sz  is the umber of symmetry entries.
%   - sets.tbs is the Thread Block Size
%   - sets.numSMs is the number of multiprocessors on the device
%   - sets.WarpSize is the warp size
%
%   See also STIFFMA_VPS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 28/01/2020. Version: 1.4. Name changed, Doc improved
% 	Modified: 21/01/2019. Version: 1.3
%   Created:  17/01/2019. Version: 1.0

% MATLAB KERNEL CREATION
if strcmp(sets.dTE,'uint32')               % uint32
    ker = parallel.gpu.CUDAKernel('Index_vps.ptx',...                           % PTXFILE
        'const unsigned int*,const unsigned int,unsigned int*,unsigned int*',...% C prototype for kernel
        'IndexVectorGPUIj');                                                    % Specify entry point
elseif strcmp(sets.dTE,'uint64')           % uint64
    ker = parallel.gpu.CUDAKernel('Index_vps.ptx',...
        'const unsigned long *, const unsigned long, unsigned long *, unsigned long *',...
        'IndexVectorGPUIm');
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
ker.ThreadBlockSize = [sets.tbs, 1, 1];                   	% Threads per block
ker.GridSize = [sets.WarpSize*sets.numSMs, 1, 1];           % Blocks per grid

% INITIALIZATION OF GPU VARIABLES
iK  = zeros(sets.sz*sets.nel, 1, sets.dTE, 'gpuArray');     % Stores row indices (initialized directly on GPU)
jK  = zeros(sets.sz*sets.nel, 1, sets.dTE, 'gpuArray');     % Stores column indices (initialized directly on GPU)

% MATLAB KERNEL CALL
[iK, jK] = feval(ker, elements, sets.nel, iK, jK);         	% GPU code execution
