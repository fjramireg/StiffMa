
% Script to run the HEX8 (ke) code on the CPU
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
settings.dTN = dTypeN;           	% Data precision for computing
settings.nel = size(elements,1);	% Number of finite elements

%% Element stiffness matrix computation on CPU (symmetry)
tic;
Keh = Hex8scalarsas(elements, nodes, c, settings);   % NNZ entries of tril(K)
time_h = toc;
fprintf('Elapsed time for computing the element stiffness matrices on serial CPU: %f\n',time_h);
