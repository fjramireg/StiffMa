
% Script to run the HEX8 (ke) code on the CPU
addpath('../Common');
addpath('../Utils');

%% Problem setup
c = 1.0;            % Conductivity (homogeneous, linear, isotropic material)
nelx = 10;          % Number of elements on X-direction
nely = 10;          % Number of elements on Y-direction
nelz = 10 ;         % Number of elements on Z-direction
dTypeE = 'uint32';  % Data precision for "elements" ['uint32', 'uint64' or 'double']
dTypeN = 'single';  % Data precision for "nodes" ['single' or 'double']

%% Mesh generation
[elements, nodes] = CreateMesh(nelx,nely,nelz,dTypeE,dTypeN);

%% Element stiffness matrix computation on CPU (symmetry)
tic;
Keh = Hex8scalarsas(elements,nodes,c,dTypeN);   % NNZ entries of tril(K)
time_h = toc;
fprintf('Time spend computing the element stiffness matrices on serial CPU: %f\n',time_h);
