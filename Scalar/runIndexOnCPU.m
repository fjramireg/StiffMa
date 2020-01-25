
% Script to run the INDEX code alone on the CPU
addpath('../Common');
addpath('../Utils');

%% Problem setup
c = 1.0;            % Conductivity (homogeneous, linear, isotropic material)
nelx = 10;          % Number of elements on X-direction
nely = 10;          % Number of elements on Y-direction
nelz = 10;          % Number of elements on Z-direction
dTypeE = 'uint32';  % Data precision for "elements" ['uint32', 'uint64']

%% Mesh generation
[elements, nodes] = CreateMesh(nelx,nely,nelz,dTypeE,'single');

%% Settings
settings.dType = dTypeE;            % Data precision for computing
settings.nel   = size(elements,1);  % Number of finite elements

%% Index computation on CPU (symmetry)
tic;
[iKh, jKh] = IndexScalarsas(elements, settings);	% Row/column indices of tril(K)
times = toc;
fprintf('Elapsed time for computing row/column indices of tril(K) on serial CPU: %f\n',times);
