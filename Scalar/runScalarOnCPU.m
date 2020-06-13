% Runs the whole assembly scalar code on the CPU
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
[Mesh.elements, Mesh.nodes] = CreateMesh2(nelx,nely,nelz,dTE,dTN);
[nel, nxe] = size(Mesh.elements);
[nnod, dim] = size(Mesh.nodes);

%% Material properties
c = 1.0;            % Conductivity (homogeneous, linear, isotropic material)

%% Settings
sets.dTE = dTE;         % Data precision for computing
sets.dTN = dTN;         % Data precision for computing
sets.nel = nel;         % Number of finite elements
sets.nxe = nxe;         % Number of nodes per element
sets.dxn = dxn;         % Number of DOFs per node 
sets.edof= dxn*nxe;     % Number of DOFs per element 
sets.tdofs = nnod;        % Number of total DOFs
sets.sz  = sets.edof * (sets.edof + 1) / 2; % Number of symmetry entries

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
