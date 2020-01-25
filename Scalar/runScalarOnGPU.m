
% run the whole assembly code on the GPU
addpath('../Common');
addpath('../Utils');

%% Problem setup
c = 1.0;            % Conductivity (homogeneous, linear, isotropic material)
nelx = 10;          % Number of elements on X-direction
nely = 10;          % Number of elements on Y-direction
nelz = 10 ;         % Number of elements on Z-direction
dTypeE = 'uint32';  % Data precision for "elements" ['uint32', 'uint64']
dTypeN = 'double';  % Data precision for "nodes" ['single' or 'double']

%% Mesh generation
[elements, nodes] = CreateMesh(nelx,nely,nelz,dTypeE,dTypeN);

%% Settings
d = gpuDevice;
settings.dTE      = dTypeE;                 % Data precision for computing
settings.dTN      = dTypeN;                 % Data precision for computing
settings.tbs      = 1024;                   % Max. Thread Block Size
settings.nel      = size(elements,1);       % Number of finite elements
settings.numSMs   = d.MultiprocessorCount;  % Number of multiprocessors on the device
settings.WarpSize = d.SIMDWidth;            % The warp size in threads

%% Creation of global stiffness matrix on GPU
tic;
elementsGPU = gpuArray(elements');          % Transfer transposed array to GPU memory
nodesGPU    = gpuArray(nodes');             % Transfer transposed array to GPU memory
K = StiffMaps(elementsGPU, nodesGPU, c, settings);% Generate the stiffness matrix on GPU (tril(K))
wait(d);
times = toc;
fprintf('Elapsed time for building tril(K) on parallel GPU: %f\n',times);
