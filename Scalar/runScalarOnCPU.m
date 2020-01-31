
% run the whole assembly code on the CPU
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
settings.dTE = dTypeE;             	% Data precision for computing
settings.dTN = dTypeN;           	% Data precision for computing
settings.nel = size(elements,1);	% Number of finite elements

%% Creation of global stiffness matrix on CPU (serial)
tic;
K = StiffMas(elements, nodes, c, settings);     % Assembly on CPU
time = toc;
fprintf('\nElapsed time for building K on serial CPU: %f\n',time);

%% Creation of global stiffness matrix on CPU (serial) taking advantage of symmetry
tic;
K_s = StiffMass(elements, nodes, c, settings);  % Assembly on CPU (tril(K))
times = toc;
fprintf('Elapsed time for building tril(K) on serial CPU: %f\n',times);