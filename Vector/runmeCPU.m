% run the whole assembly code on the CPU

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
E = 200e9;          % Elastic modulus (homogeneous, linear, isotropic material)
nu = 0.3;           % Poisson ratio

%% Global stiffness matrix creation
K_h = AssemblyVector(elements,nodes,E,nu);      % Assembly on the CPU
K_s = AssemblyVectorSym(elements,nodes,E,nu);   % Assembly on the CPU (Symmetric matrix)
