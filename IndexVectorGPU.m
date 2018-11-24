function [iK, jK] = IndexVectorGPU(elements)
% Row/column indices of the lower triangular sparse matrix K (VECTOR)

% INITIALIZATION OF GPU VARIABLES
elements = gpuArray(uint32(elements));   % Converts the data precision and transfer it to GPU
elements = sort(elements,2);             % Sort the nodes within the connectivity matrix
nel = size(elements,1);                  % Number of elements
iK  = gpuArray.zeros(300*nel,1,'uint32'); % Stores row indices (initialized directly on GPU)
jK  = gpuArray.zeros(300*nel,1,'uint32'); % Stores column indices (initialized directly on GPU)

% MATLAB KERNEL CREATION
IdXkernel = parallel.gpu.CUDAKernel('IndexVectorGPU.ptx', 'IndexVectorGPU.cu');

% MATLAB KERNEL CONFIGURATION
IdXkernel.ThreadBlockSize = [512, 1, 1];                                % Threads per block
IdXkernel.GridSize = [ceil(nel/IdXkernel.ThreadBlockSize(1)), 1, 1];    % Blocks per grid

% MATLAB KERNEL CALL
[iK, jK] = feval(IdXkernel, elements', nel, iK, jK);                    % GPU code execution
