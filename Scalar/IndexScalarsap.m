function [iK, jK] = IndexScalarsap(elements, tbs)
% INDEXSCALARSAP Compute the row/column indices of tril(K) in PARALLEL computing
% for a SCALAR problem taking advantage of GPU computing.
%   INDEXSCALARSAP(elements) returns the rows "iK" and columns "jK" position
%   of all element stiffness matrices in the global system for a finite element
%   analysis of a scalar problem in a three-dimensional domain taking advantage
%   of symmetry and GPU computing, where "elements" is the connectivity matrix
%   of size 8xnel and the optional "tbs" refers to ThreadBlockSize (scalar).
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

dType = classUnderlying(elements);          % Data type (uint32, uint64, double)
nel = size(elements,2);                     % Number of elements

% MATLAB KERNEL CREATION
if strcmp(dType,'uint32')                   % uint32
    ker = parallel.gpu.CUDAKernel('IndexScalarsp.ptx',...                       % PTXFILE
        'const unsigned int*, const unsigned int, unsigned int*, unsigned int*',... % C prototype for kernel
        'IndexScalarGPUIj');                                                    % Specify entry point
elseif strcmp(dType,'uint64')               % uint64
    ker = parallel.gpu.CUDAKernel('IndexScalarsp.ptx',...
        'const unsigned long *, const unsigned long, unsigned long *, unsigned long *',...
        'IndexScalarGPUIm');
elseif strcmp(dType,'double')               % double
    ker = parallel.gpu.CUDAKernel('IndexScalarsp.ptx',...
        'const double *, const double, double *, double *',...
        'IndexScalarGPUId');
else
    error('Not supported data type. Use only one of this: uint32, uint64, double');
end

% MATLAB KERNEL CONFIGURATION
if (nargin == 1 || tbs > ker.MaxThreadsPerBlock)
    tbs = ker.MaxThreadsPerBlock;                        % Default (MaxThreadsPerBlock)
end
ker.ThreadBlockSize = [tbs, 1, 1];                       % Threads per block
ker.GridSize = [ceil(nel/ker.ThreadBlockSize(1)), 1, 1]; % Blocks per grid

% INITIALIZATION OF GPU VARIABLES
iK  = zeros(36*nel,1,dType,'gpuArray');                  % Stores row indices (initialized directly on GPU)
jK  = zeros(36*nel,1,dType,'gpuArray');                  % Stores column indices (initialized directly on GPU)

% MATLAB KERNEL CALL
[iK, jK] = feval(ker, elements, nel, iK, jK);            % GPU code execution
