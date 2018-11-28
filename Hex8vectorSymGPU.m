function KE = Hex8vectorSymGPU(elements,nodes,E,nu)
% Symmetric part of the element stiffness matrix ke (VECTOR-DOUBLE)

% FUNCTION CALLS
L = dNdrst;
D = MaterialMatrix(E,nu);

% INITIALIZATION OF GPU VARIABLES
elements = gpuArray(uint32(elements));   % Converts the data precision and transfer it to GPU
nodes = gpuArray(nodes);                 % Transfer to GPU memory
% L = gpuArray(L);                         % Transfer to GPU memory
% D = gpuArray(D);                         % Transfer to GPU memory 
nel = size(elements,1);                  % Number of elements
nnod = size(nodes,1);                    % Number of nodes
KE = gpuArray.zeros(300*nel,1,'double'); % Stores ke entries (initialized directly on GPU)

% MATLAB KERNEL CREATION
kernel = parallel.gpu.CUDAKernel('Hex8vectorSymGPU.ptx', 'Hex8vectorSymGPU.cu');

% MATLAB KERNEL CONFIGURATION
kernel.ThreadBlockSize = [256, 1, 1];                             % Threads per block
kernel.GridSize = [ceil(nel/kernel.ThreadBlockSize(1)), 1, 1];    % Blocks per grid

% MATLAB KERNEL CALL
KE = feval(kernel, elements', nodes', nel, nnod, L, D, KE);       % GPU code execution
