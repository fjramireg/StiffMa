
% Script to run the INDEX code on GPU
addpath('../Common');
addpath('../Utils');

%% Problem setup
c = 1.0;            % Conductivity (homogeneous, linear, isotropic material)
nelx = 100;         % Number of elements on X-direction
nely = 100;         % Number of elements on Y-direction
nelz = 100;         % Number of elements on Z-direction
dTypeE = 'uint32';  % Data precision for "elements" ['uint32', 'uint64' or 'double']

%% Mesh generation
[elements, nodes] = CreateMesh(nelx,nely,nelz,dTypeE,'single');

%% Index computation on GPU (symmetry)
d = gpuDevice;
elementsGPU = gpuArray(elements');          % Transfer transposed array to GPU memory
tbs = 256;                                  % Thread Block Size
tic;
[iKd, jKd] = IndexScalarsap(elementsGPU, dTypeE, tbs); % Row/column indices of tril(K)
wait(d);
times = toc;
fprintf('Time spend computing row/column indices of tril(K) on parallel GPU: %f\n',times);
