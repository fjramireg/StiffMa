
% run the whole assembly code on the CPU
addpath('../Common');
addpath('../Utils');

%% Problem setup
c = 1.0;            % Conductivity (homogeneous, linear, isotropic material)
nelx = 10;          % Number of elements on X-direction
nely = 10;          % Number of elements on Y-direction
nelz = 10 ;         % Number of elements on Z-direction
dTypeE = 'uint32';  % Data precision for "elements" ['int32', 'uint32', 'int64', 'uint64' or 'double']
dTypeN = 'double';  % Data precision for "nodes" ['single' or 'double']

%% Mesh generation
[elements, nodes] = CreateMesh(nelx,nely,nelz,dTypeE,dTypeN);

%% Creation of global stiffness matrix taking advantage of symmetry
tic;
K_s = StiffMass(elements,nodes,c);  % Assembly on CPU (tril(K))
times = toc;
fprintf('Time spend to build tril(K) on serial CPU: %f\n',times);

tic;
K_2 = StiffMass_mex(elements,nodes,c);  % Assembly on GPU (tril(K)) using GPU Coder
times = toc;
fprintf('Time spend to build tril(K) on GPU Coder: %f\n',times);
