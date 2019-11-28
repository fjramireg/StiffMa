function [iK, jK] = IndexScalarSymGPU(elements)
% INDEXSCALARSYMGPU Compute the row and column indices of lower symmetric
% part of global stiffness matrix for a SCALAR problem taking advantage of
% GPU computing.
%   INDEXSCALARSYMGPU(elements) returns the rows "iK" and columns "jK" position
%   of all element stiffness matrices in the global system for a finite
%   element analysis of a scalar problem in a three-dimensional domain
%   taking advantage of symmetry and GPU computing, where "elements" is the
%   connectivity matrix.
%
%   See also INDEXSCALARSYMCPU, INDEXSCALARSYMCPUP, STIFFMATGENSCSYMGPU
%
%   For more information, see <a href="matlab:
%   web('https://github.com/fjramireg/MatGen')">the MatGen Web site</a>.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created: 30/11/2018. Modified: 21/01/2019. Version: 1.3

dType = classUnderlying(elements);          % Data type (int32, uint32, int64, uint64, double)
nel = size(elements,2);                     % Number of elements

% INITIALIZATION OF GPU VARIABLES
iK  = zeros(36*nel,1,dType,'gpuArray');     % Stores row indices (initialized directly on GPU)
jK  = zeros(36*nel,1,dType,'gpuArray');     % Stores column indices (initialized directly on GPU)

% MATLAB KERNEL CREATION
if strcmp(dType,'int32')                    % int32
    ker = parallel.gpu.CUDAKernel('IndexScalarGPU.ptx',...  % PTXFILE
        'const int *, const int, int *, int *',...          % C prototype for kernel
        'IndexScalarGPUIi');                                % Specify entry point
elseif strcmp(dType,'uint32')               % uint32
    ker = parallel.gpu.CUDAKernel('IndexScalarGPU.ptx',...
        'const unsigned int *, const unsigned int, unsigned int *, unsigned int *',...
        'IndexScalarGPUIj');
elseif strcmp(dType,'int64')                % int64
    ker = parallel.gpu.CUDAKernel('IndexScalarGPU.ptx',...
        'const long *, const long, long *, long *',...
        'IndexScalarGPUIl');
elseif strcmp(dType,'uint64')               % uint64
    ker = parallel.gpu.CUDAKernel('IndexScalarGPU.ptx',...
        'const unsigned long *, const unsigned long, unsigned long *, unsigned long *',...
        'IndexScalarGPUIm');
elseif strcmp(dType,'double')               % double
    ker = parallel.gpu.CUDAKernel('IndexScalarGPU.ptx',...
        'const double *, const double, double *, double *',...
        'IndexScalarGPUId');
else
    error('Not supported data type. Use only one of this: int32, uint32, int64, uint64, double');
end

% MATLAB KERNEL CONFIGURATION
ker.ThreadBlockSize = [ker.MaxThreadsPerBlock, 1, 1];    % Threads per block
ker.GridSize = [ceil(nel/ker.ThreadBlockSize(1)), 1, 1]; % Blocks per grid

% MATLAB KERNEL CALL
[iK, jK] = feval(ker, elements, nel, iK, jK);            % GPU code execution
