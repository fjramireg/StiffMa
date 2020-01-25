function [iK, jK] = IndexScalarsap_smem(elements, settings)
% INDEXSCALARSAP Compute the row/column indices of tril(K) in PARALLEL computing
% for a SCALAR problem taking advantage of GPU computing.
%   INDEXSCALARSAP(elements, tbs, dType) returns the rows "iK" and columns "jK"
%   position of all element stiffness matrices in the global system for a finite
%   element analysis of a scalar problem in a three-dimensional domain taking
%   advantage of symmetry and GPU computing, where "elements" is the
%   connectivity matrix of size 8xnel and dType is the data type defined to the
%   "elements" array. The optional "tbs" refers to ThreadBlockSize (scalar).
%
%   See also STIFFMAPS, INDEXSCALARSAS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 03/12/2019. Version: 1.4. Variable number of inputs
% 	Modified: 21/01/2019. Version: 1.3
%   Created:  30/11/2018. Version: 1.0

% MATLAB KERNEL CREATION
if strcmp(settings.dTE,'uint32')                      % uint32
    ker = parallel.gpu.CUDAKernel('IndexScalarsp_smem.ptx',...                       % PTXFILE
        'const unsigned int*,const unsigned int,unsigned int*,unsigned int*',...% C prototype for kernel
        'IndexScalarGPUIj');                                                    % Specify entry point
elseif strcmp(settings.dTE,'uint64')                  % uint64
    ker = parallel.gpu.CUDAKernel('IndexScalarsp_smem.ptx',...
        'const unsigned long *, const unsigned long, unsigned long *, unsigned long *',...
        'IndexScalarGPUIm');
else
    error('Not supported data type. Use only one of this: uint32, uint64');
end

% MATLAB KERNEL CONFIGURATION
if (settings.tbs > ker.MaxThreadsPerBlock || mod(settings.tbs, settings.WarpSize) )
    settings.tbs = ker.MaxThreadsPerBlock;
    if  mod(settings.tbs, settings.WarpSize)
        settings.tbs = settings.tbs - mod(settings.tbs, settings.WarpSize);
    end
end
ker.ThreadBlockSize = [settings.tbs, 1, 1];                   	% Threads per block
ker.GridSize = [settings.WarpSize*settings.numSMs, 1, 1];       % Blocks per grid   
% ker.GridSize = [ceil(settings.nel/settings.tbs), 1, 1];       % Blocks per grid  

% INITIALIZATION OF GPU VARIABLES
iK  = zeros(36*settings.nel, 1, settings.dTE, 'gpuArray');    % Stores row indices (initialized directly on GPU)
jK  = zeros(36*settings.nel, 1, settings.dTE, 'gpuArray');	% Stores column indices (initialized directly on GPU)

% MATLAB KERNEL CALL
[iK, jK] = feval(ker, elements, settings.nel, iK, jK);        	% GPU code execution
