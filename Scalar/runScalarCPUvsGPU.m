
% Script to run the whole assembly code on the CPU and GPU, and compare them
addpath('../Common');
addpath('../Utils');

%% Problem setup
c = 1.0;            % Conductivity (homogeneous, linear, isotropic material)
nelx = 10;          % Number of elements on X-direction
nely = 10;          % Number of elements on Y-direction
nelz = 10 ;         % Number of elements on Z-direction
dTypeE = 'uint32';  % Data precision for "elements" ['uint32', 'uint64' or 'double']
dTypeN = 'double';  % Data precision for "nodes" ['single' or 'double']

%% Mesh generation
[elements, nodes] = CreateMesh(nelx,nely,nelz,dTypeE,dTypeN);

%% Creation of global stiffness matrix on CPU (serial)
tic;
K = StiffMas(elements,nodes,c);     % Assembly on CPU
time = toc;
fprintf('\nTime spend building K on serial CPU: %f\n',time);

%% Creation of global stiffness matrix on CPU (serial) taking advantage of symmetry
tic;
Ks = StiffMass(elements,nodes,c);  % Assembly on CPU (tril(K))
time_h = toc;
fprintf('Time spend building tril(K) on serial CPU: %f\n',time_h);

%% Creation of global stiffness matrix on GPU (parallel) taking advantage of symmetry
d = gpuDevice;
tbs = 256;                                      % Thread Block Size
elementsGPU = gpuArray(elements');              % Transfer transposed array to GPU memory
nodesGPU    = gpuArray(nodes');                 % Transfer transposed array to GPU memory
tic;
Ks_d = StiffMaps(elementsGPU,nodesGPU,c,tbs);   % Generate the stiffness matrix on GPU (tril(K))
wait(d);
time_d = toc;
fprintf('Time spend building tril(K) on parallel GPU: %f\n',time_d);
fprintf('GPU speedup: %f\n',time_h/time_d);

%% Difference between results
fprintf('Difference between results:\n');
K2 = tril(K);
fprintf('\tCPU vs CPU (K vs tril(K)): %e\n',norm(K2(:)-Ks(:)));
Ks_d2 = gather(Ks_d);
fprintf('\tCPU vs GPU (tril(K) vs tril(K)): %e\n',norm(Ks(:)-Ks_d2(:)));
