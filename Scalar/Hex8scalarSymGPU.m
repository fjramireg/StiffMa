%  * ====================================================================*/
% ** This function was developed by:
%  *          Francisco Javier Ramirez-Gil
%  *          Universidad Nacional de Colombia - Medellin
%  *          Department of Mechanical Engineering
%  *
%  ** Please cite this code as:
%  *
%  ** Date & version
%  *      30/11/2018.
%  *      V 1.2
%  *
%  * ====================================================================*/

function KE = Hex8scalarSymGPU(elements,nodes,c)
% Symmetric part of the element stiffness matrix ke (SCALAR-DOUBLE)

% INITIALIZATION OF GPU VARIABLES
elements = gpuArray(uint32(elements));   % Converts the data precision and transfer it to GPU
nodes = gpuArray(nodes);                 % Transfer to GPU memory
L = gpuArray(dNdrst);                    % % Transfer to GPU memory
nel = size(elements,1);                  % Number of elements
nnod = size(nodes,1);                    % Number of nodes
KE = gpuArray.zeros(36*nel,1,'double');  % Stores ke entries (initialized directly on GPU)

% MATLAB KERNEL CREATION
kernel = parallel.gpu.CUDAKernel('Hex8scalarSymGPU.ptx', 'Hex8scalarSymGPU.cu');

% MATLAB KERNEL CONFIGURATION
kernel.ThreadBlockSize = [256, 1, 1];                             % Threads per block
kernel.GridSize = [ceil(nel/kernel.ThreadBlockSize(1)), 1, 1];    % Blocks per grid

% MATLAB KERNEL CALL
KE = feval(kernel, elements', nodes', nel, nnod, L, c, KE);       % GPU code execution
