function [iK, jK] = IndexScalarGPU(elements)
% Row/column indices of the lower triangular part of the sparse stiffness matrix K (SCALAR)

% INITIALIZATION OF GPU VARIABLES
elements = gpuArray(uint32(elements));   % Converts the data precision and transfer it to GPU
nel = size(elements,1);                  % Number of elements
iK  = gpuArray.zeros(36*nel,1,'uint32'); % Stores row indices (initialized directly on GPU)
jK  = gpuArray.zeros(36*nel,1,'uint32'); % Stores column indices (initialized directly on GPU)

% MATLAB KERNEL CREATION
IdXkernel = parallel.gpu.CUDAKernel('IndexScalarGPU.ptx', 'IndexScalarGPU.cu');

% MATLAB KERNEL CONFIGURATION
IdXkernel.ThreadBlockSize = [512, 1, 1];                                % Threads per block
IdXkernel.GridSize = [ceil(nel/IdXkernel.ThreadBlockSize(1)), 1, 1];    % Blocks per grid 

% MATLAB KERNEL CALL
[iK, jK] = feval(IdXkernel, elements', nel, iK, jK);                    % GPU code execution
