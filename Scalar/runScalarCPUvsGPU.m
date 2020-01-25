
% Script to run the whole assembly code on the CPU and GPU, and compare them
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

%% Creation of global stiffness matrix on CPU (serial)
tic;
K = StiffMas(elements, nodes, c, settings);     % Assembly on CPU
time = toc;
fprintf('\nElapsed time for building K on serial CPU: %f\n',time);

%% Creation of global stiffness matrix on CPU (serial) taking advantage of symmetry
tic;
Ks_h = StiffMass(elements, nodes, c, settings);  % Assembly on CPU (tril(K))
time_h = toc;
fprintf('Elapsed time for building tril(K) on serial CPU: %f\n',time_h);
fprintf('\tCPU speedup (CPU vs CPU, K vs tril(K)): %f\n',time/time_h);

%% Creation of global stiffness matrix on GPU (parallel) taking advantage of symmetry
elementsGPU = gpuArray(elements');              % Transfer transposed array to GPU memory
nodesGPU    = gpuArray(nodes');                 % Transfer transposed array to GPU memory
tic;
Ks_d = StiffMaps(elementsGPU, nodesGPU, c, settings);   % Generate the stiffness matrix on GPU (tril(K))
wait(d);
time_d = toc;
fprintf('Elapsed time for building tril(K) on parallel GPU: %f\n',time_d);
fprintf('\tGPU speedup (K vs tril(K)): %f\n',time/time_d);
fprintf('\tGPU speedup (tril(K) vs tril(K)): %f\n',time_h/time_d);

%% Difference between results
fprintf('Difference between results:\n');
K2 = tril(K);
fprintf('\tCPU vs CPU (K vs tril(K)): %e\n',norm(K2(:)-Ks_h(:)));
Ks_d2 = gather(Ks_d);
fprintf('\tCPU vs GPU (tril(K) vs tril(K)): %e\n',norm(Ks_h(:)-Ks_d2(:)));
