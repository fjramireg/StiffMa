
% Script to run the HEX8 (ke) code on the CPU and GPU, and compare them
addpath('../Common');
addpath('../Utils');

%% Problem setup
c = 1.0;            % Conductivity (homogeneous, linear, isotropic material)
nelx = 100;          % Number of elements on X-direction
nely = 10;          % Number of elements on Y-direction
nelz = 10 ;         % Number of elements on Z-direction
dTypeE = 'uint32';  % Data precision for "elements" ['uint32', 'uint64' or 'double']
dTypeN = 'double';  % Data precision for "nodes" ['single' or 'double']

%% Mesh generation
[elements, nodes] = CreateMesh(nelx,nely,nelz,dTypeE,dTypeN);

%% Element stiffness matrix computation on CPU (symmetry)
tic;
Keh = Hex8scalarsas(elements,nodes,c,dTypeN);% NNZ entries of tril(K)
time_h = toc;
fprintf('Elapsed time for computing the element stiffness matrices on serial CPU: %f\n',time_h);

%% Element stiffness matrix computation on GPU (symmetry)
d = gpuDevice;
elements_d = gpuArray(elements');           % Transfer transposed array to GPU memory
nodes_d = gpuArray(nodes');                 % Transfer transposed array to GPU memory
tbs = 256;                                  % Thread Block Size
tic;
Ked = Hex8scalarsap(elements_d,nodes_d,c,dTypeE,dTypeN,tbs);% NNZ entries of tril(K)
wait(d);
time_d = toc;
fprintf('Elapsed time for computing the element stiffness matrices on parallel GPU: %f\n',time_d);
fprintf('GPU speedup: %f\n',time_h/time_d);

%% Difference between results
fprintf('Difference between results:\n');
fprintf('\tCPU vs GPU (Ked vs iKd): %e\n',norm(Keh(:)-Ked(:)));
