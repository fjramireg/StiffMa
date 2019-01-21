%  * ====================================================================*/
% ** This function was developed by:
%  *          Francisco Javier Ramirez-Gil
%  *          Universidad Nacional de Colombia - Medellin
%  *          Department of Mechanical Engineering
%  *
%  ** Please cite this code as:
%  *
%  ** Date & version
%  *      17/01/2019.
%  *      V 1.2
%  *
%  * ====================================================================*/

function [iK, jK] = IndexVectorSymGPU(elements)
% Row/column indices of the lower triangular part of the sparse stiffness matrix K (VECTOR)

dType = classUnderlying(elements);          % Data type (int32, uint32, int64, uint64, double)
nel = size(elements,2);                     % Number of elements

% INITIALIZATION OF GPU VARIABLES
iK  = zeros(300*nel,1,dType,'gpuArray');    % Stores row indices (initialized directly on GPU)
jK  = zeros(300*nel,1,dType,'gpuArray');    % Stores column indices (initialized directly on GPU)

% MATLAB KERNEL CREATION
if strcmp(dType,'int32')                    % int32
    ker = parallel.gpu.CUDAKernel('IndexVectorGPU.ptx',...  % PTXFILE
        'const int *, const int, int *, int *',...          % C prototype for kernel
        'IndexVectorGPUIi');                                % Specify entry point
elseif strcmp(dType,'uint32')               % uint32
    ker = parallel.gpu.CUDAKernel('IndexVectorGPU.ptx',...
        'const unsigned int *, const unsigned int, unsigned int *, unsigned int *',...
        'IndexVectorGPUIj');
elseif strcmp(dType,'int64')                % int64
    ker = parallel.gpu.CUDAKernel('IndexVectorGPU.ptx',...
        'const long *, const long, long *, long *',...
        'IndexVectorGPUIl');
elseif strcmp(dType,'uint64')               % uint64
    ker = parallel.gpu.CUDAKernel('IndexVectorGPU.ptx',...
        'const unsigned long *, const unsigned long, unsigned long *, unsigned long *',...
        'IndexVectorGPUIm');
elseif strcmp(dType,'double')               % double
    ker = parallel.gpu.CUDAKernel('IndexVectorGPU.ptx',...
        'const double *, const double, double *, double *',...
        'IndexVectorGPUId');
else
    error('Not supported data type. Use only one of this: int32, uint32, int64, uint64, double');
end

% MATLAB KERNEL CONFIGURATION
ker.ThreadBlockSize = [ker.MaxThreadsPerBlock, 1, 1];    % Threads per block
ker.GridSize = [ceil(nel/ker.ThreadBlockSize(1)), 1, 1]; % Blocks per grid

% MATLAB KERNEL CALL
[iK, jK] = feval(ker, elements, nel, iK, jK);            % GPU code execution
