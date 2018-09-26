function [iK, jK] = IndexScalarGPU(elements)
% Computation of the indices of the sparse matrix
elements = gpuArray(uint32(elements));   % Converts the precision data and transfer to GPU
nel = size(elements,1);                  % Number of FEs
iK  = gpuArray.zeros(36*nel,1,'uint32'); % Store row indices (initialized directly on GPU)
jK  = gpuArray.zeros(36*nel,1,'uint32'); % Store column indices (initialized directly on GPU)

% MATLAB KERNEL CREATION
IdXkernel = parallel.gpu.CUDAKernel('IndexScalarGPU.ptx', 'IndexScalarGPU.cu');

% MATLAB KERNEL CONFIGURATION
IdXkernel.ThreadBlockSize = [512, 1, 1];
IdXkernel.GridSize = [ceil(nel/IdXkernel.ThreadBlockSize(1)), 1, 1];

% MATLAB CALL
 [iK, jK] = feval(IdXkernel, elements', nel, iK, jK);
 