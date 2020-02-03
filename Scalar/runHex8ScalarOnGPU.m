% Runs the HEX8 scalar code on the GPU
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.
%
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  31/01/2020. Version: 1.4

%% Add some common paths
addpath('../Common');
addpath('../Utils');

%% Mesh generation
dxn = 1;            % For vector 3 (UX, UY, UZ). For scalar 1 (Temp)
nelx = 10;          % Number of elements on X-direction
nely = 10;          % Number of elements on Y-direction
nelz = 10;          % Number of elements on Z-direction
dTE = 'uint32';     % Data precision for "elements" ['uint32', 'uint64']
dTN = 'double';     % Data precision for "nodes" ['single' or 'double']
[elements, nodes] = CreateMesh(nelx,nely,nelz,dTE,dTN);
[nel, nxe] = size(elements);

%% Material properties
c = 1.0;            % Conductivity (homogeneous, linear, isotropic material)

%% General Settings
sets.dTE = dTE;     % Data precision for computing
sets.dTN = dTN;     % Data precision for computing
sets.nel = nel;     % Number of finite elements
sets.nxe = nxe;     % Number of nodes per element
sets.dxn = dxn;     % Number of DOFs per node
sets.edof= dxn*nxe; % Number of DOFs per element
sets.sz  = sets.edof * (sets.edof + 1) / 2; % Number of symmetry entries

%% GPU Settings
d = gpuDevice;
sets.tbs      = d.MaxThreadsPerBlock;   % Max. Thread Block Size
sets.numSMs   = d.MultiprocessorCount;  % Number of multiprocessors on the device
sets.WarpSize = d.SIMDWidth;            % The warp size in threads

%% Element stiffness matrix computation on GPU (symmetry)
tic;
elementsGPU = gpuArray(elements');      % Transfer transposed array to GPU memory
nodesGPU = gpuArray(nodes');            % Transfer transposed array to GPU memory
Ke  = eStiff_spsa(elementsGPU, nodesGPU, c, sets); % Computation of Ke for tril(K)
wait(d);
times = toc;
fprintf('Elapsed time for computing row/column indices of tril(K) on parallel GPU: %f\n',times);
