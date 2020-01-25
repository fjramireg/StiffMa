
% Script to run the INDEX code alone on the GPU
addpath('../Common');
addpath('../Utils');

%% Problem setup
c = 1.0;            % Conductivity (homogeneous, linear, isotropic material)
nelx = 10;         % Number of elements on X-direction
nely = 10;         % Number of elements on Y-direction
nelz = 10;         % Number of elements on Z-direction
dTypeE = 'uint32';  % Data precision for "elements" ['uint32', 'uint64']

%% Mesh generation
[elements, ~] = CreateMesh(nelx,nely,nelz,dTypeE,'single');

%% Settings
% d = gpuDevice;
% settings.dTE      = dTypeE;                 % Data precision for computing
% settings.tbs      = 1024;                   % Max. Thread Block Size
% settings.nel      = size(elements,1);       % Number of finite elements
% settings.numSMs   = d.MultiprocessorCount;  % Number of multiprocessors on the device
% settings.WarpSize = d.SIMDWidth;            % The warp size in threads
% 
% %% Index computation on GPU (symmetry)
% tic;
% elementsGPU = gpuArray(elements');                  % Transfer transposed array to GPU memory
% [iKd, jKd]  = IndexScalarsap_smem(elementsGPU, settings);% Computation of row/column indices of tril(K)
% wait(d);
% times = toc;
% fprintf('Elapsed time for computing row/column indices of tril(K) on parallel GPU: %f\n',times);
