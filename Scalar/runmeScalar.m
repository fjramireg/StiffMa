% run the whole assembly code on the CPU
addpath('../Common');

%% Mesh generation
nelx = 3;           % Number of elements on X-direction
nely = 2;           % Number of elements on Y-direction
nelz = 1;           % Number of elements on Z-direction
dTypeE = 'uint32';  % Data precision for "elements" ['int32', 'uint32', 'int64', 'uint64' or 'double']
dTypeN = 'double';  % Data precision for "nodes" ['single' or 'double']
PlotE = 0;          % Plot the elements and their numbers (1 to plot)
PlotN = 0;          % Plot the nodes and their numbers (1 to plot)
[elements, nodes] = CreateMesh(nelx,nely,nelz,dTypeE,dTypeN,PlotE,PlotN);

%% Material properties
c = 1;              % Conductivity (homogeneous, linear, isotropic material)

%% Creation of global stiffness matrix on CPU
K_h = AssemblyScalar(elements,nodes,c);             % Assembly on CPU
K_s = AssemblyScalarSym(elements,nodes,c);          % Assembly on CPU (symmetric)

%% Creation of global stiffness matrix on GPU
elementsGPU = gpuArray(elements');                  % Transfer to GPU memory
nodesGPU    = gpuArray(nodes');                     % Transfer to GPU memory
K_d = AssemblyScalarSymGPU(elementsGPU,nodesGPU,c); % Assembly on GPU (symmetric)
