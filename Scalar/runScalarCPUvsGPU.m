% Runs the whole assembly scalar. CPU vs GPU
%
% This script is used to generate the global stiffness matrix K for the VECTOT
% problem (linear static elasticiy).
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.
%
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  28/01/2020. Version: 1.4

% run the whole assembly code on the CPU and GPU
addpath('../Common');
addpath('../Utils');

%% Mesh generation
nelx = 10;          % Number of elements on X-direction
nely = 10;          % Number of elements on Y-direction
nelz = 10;          % Number of elements on Z-direction
dTE = 'uint64';     % Data precision for "elements" ['uint32', 'uint64' or 'double']
dTN = 'double';     % Data precision for "nodes" ['single' or 'double']
PlotE = 0;          % Plot the elements and their numbers (1 to plot)
PlotN = 0;          % Plot the nodes and their numbers (1 to plot)
[Mesh.elements, Mesh.nodes] = CreateMesh2(nelx,nely,nelz,dTE,dTN);
[nel, nxe] = size(Mesh.elements);

%% Material properties
c = 1.0;            % Conductivity (homogeneous, linear, isotropic material)

%% General settings
dxn = 1;            % For vector 3 (UX, UY, UZ). For scalar 1 (Temp)
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

%% Creation of global stiffness matrix on CPU (serial)
tic;
K_f = StiffMa_ss(Mesh, c, sets);	% Assembly on CPU: K
time = toc;
fprintf('\nElapsed time for building K on serial CPU: %f\n',time);

%% Creation of global stiffness matrix on CPU (serial) taking advantage of symmetry
tic;
K_s = StiffMa_sss(Mesh, c, sets);  % Assembly on CPU (tril(K))
times = toc;
fprintf('Elapsed time for building tril(K) on serial CPU: %f\n',times);
fprintf('\tCPU speedup (CPU vs CPU, K vs tril(K)): %f\n',time/times);

%% Creation of global stiffness matrix on GPU
tic;
elementsGPU = gpuArray(Mesh.elements');	% Transfer to GPU memory
nodesGPU    = gpuArray(Mesh.nodes');  	% Transfer to GPU memory
K_d = StiffMa_sps(elementsGPU, nodesGPU, c, sets);      % Assembly on GPU (symmetric)
time_d = toc;
wait(d);
fprintf('Elapsed time for building tril(K) on parallel GPU: %f\n',time_d);
fprintf('\tGPU speedup (K vs tril(K)): %f\n',time/time_d);
fprintf('\tGPU speedup (tril(K) vs tril(K)): %f\n',times/time_d);

%% Difference between results
fprintf('Difference between results:\n');
K2 = tril(K_f);
fprintf('\tCPU vs CPU (K vs tril(K)): %e\n',norm(K2(:)-K_s(:)));
K_d2 = gather(K_d);
fprintf('\tCPU vs GPU (tril(K) vs tril(K)): %e\n',norm(K_s(:)-K_d2(:)));
