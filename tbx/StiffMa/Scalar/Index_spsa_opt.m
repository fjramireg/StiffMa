function [iK, jK] = Index_spsa_opt(elements, sets)
% INDEX_SPSA_OPT Computes the row/column indices of tril(K) for a SCALAR (s) problem
% in PARALLEL (p) GPU computing taking advantage of symmetry (s) to return ALL (a)
% indices for the mesh using some CUDA optimization (opt) techniques: 
%  1. coalesced reads/writes
%  2. __restrict__ (to helps the compiler optimize memory accesses more aggressively)
%  3. The indices should be reorganized to obtain the original MATLAB-style sz*e + t ordering, e.g.
%       iKd_col = reshape(reshape(iKd, sets.nel, sets.sz).', [], 1);
%       jKd_col = reshape(reshape(jKd, sets.nel, sets.sz).', [], 1); 
% 
%   [iK, jK]=INDEX_SPSA_OPT(elements, sets) returns the rows "iK" and columns "jK"
%   position of all element stiffness matrices in the global system for a finite
%   element analysis of a scalar problem in a three-dimensional domain taking
%   advantage of symmetry and GPU computing, where "elements" is the
%   connectivity matrix of size nelx8 and dType is the data type defined to the
%   "elements" array.  The struct "sets" must contain several simulation
%   parameters: 
%   - sets.dTE is the data precision of "Mesh.elements"
%   - sets.dTN is the data precision of "Mesh.nodes"
%   - sets.nel is the number of finite elements
%   - sets.sz  is the number of symmetry entries
%   - sets.tbs is the Thread Block Size
%   - sets.numSMs is the number of multiprocessors on the device
%   - sets.WarpSize is the warp size
%
%   See also INDEX_SPSA, INDEX_SA, STIFFMA_SPS, INDEX_SSA, INDEX_SSSA
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Institución Universitaria Pascual Bravo, Medellin-Colombia
%   Created:  April 16, 2026. Version: 1.0

% MATLAB KERNEL CREATION
if strcmp(sets.dTE,'uint32')               % uint32
    ker = parallel.gpu.CUDAKernel('Index_sps_opt.ptx',...                           % PTXFILE
        'const unsigned int*,const unsigned int,unsigned int*,unsigned int*',...% C prototype for kernel
        'IndexScalarGPUIj');                                                    % Specify entry point
elseif strcmp(sets.dTE,'uint64')           % uint64
    ker = parallel.gpu.CUDAKernel('Index_sps_opt.ptx',...
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
iK  = zeros(sets.sz*sets.nel, 1, sets.dTE, 'gpuArray');	% Stores row indices (initialized directly on GPU)
jK  = zeros(sets.sz*sets.nel, 1, sets.dTE, 'gpuArray');	% Stores column indices (initialized directly on GPU)

% MATLAB KERNEL CALL
[iK, jK] = feval(ker, elements, sets.nel, iK, jK);  % GPU code execution
