
% run the whole assembly code on the GPU
addpath('../Common');
addpath('../Utils');

%% Problem setup
tbs = 512;          % Thread Block Size
c = 1.0;            % Conductivity (homogeneous, linear, isotropic material)
nelx = 10;          % Number of elements on X-direction
nely = 10;          % Number of elements on Y-direction
nelz = 10 ;         % Number of elements on Z-direction
dTypeE = 'uint32';  % Data precision for "elements" ['uint32', 'uint64' or 'double']
dTypeN = 'double';  % Data precision for "nodes" ['single' or 'double']

%% Mesh generation
[elements, nodes] = CreateMesh(nelx,nely,nelz,dTypeE,dTypeN);

%% Creation of global stiffness matrix on GPU
d = gpuDevice;
elementsGPU = gpuArray(elements');          % Transfer transposed array to GPU memory
nodesGPU    = gpuArray(nodes');             % Transfer transposed array to GPU memory
tic;
K = StiffMaps(elementsGPU,nodesGPU,c,tbs);  % Generate the stiffness matrix on GPU (tril(K))
wait(d);
times = toc;
fprintf('Time spend building tril(K) on parallel GPU: %f\n',times);
