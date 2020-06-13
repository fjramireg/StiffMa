% Runs the whole assembly vector code on the CPU
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
dxn = 3;            % For vector 3 (UX, UY, UZ). For scalar 1 (Temp)
nelx = 10;          % Number of elements on X-direction
nely = 10;          % Number of elements on Y-direction
nelz = 10;          % Number of elements on Z-direction
dTE = 'uint32';     % Data precision for "elements" ['uint32', 'uint64']
dTN = 'double';     % Data precision for "nodes" ['single' or 'double']
[Mesh.elements, Mesh.nodes] = CreateMesh(nelx,nely,nelz,dTE,dTN);

%% Material properties
MP.E = 200e9;      	% Elastic modulus (homogeneous, linear, isotropic material)
MP.nu = 0.3;       	% Poisson ratio

%% Settings
sets.dTE = dTE;                             % Data precision for computing
sets.dTN = dTN;                             % Data precision for computing
[sets.nel, sets.nxe]  = size(Mesh.elements);% Number of elements in the mesh & Number of nodes per element
[sets.nnod, sets.dim] = size(Mesh.nodes);  	% Number of nodes in the mesh & Space dimension
sets.dxn = 3;                             	% Number of DOFs per node for the vector problem
sets.edof = sets.dxn * sets.nxe;           	% Number of DOFs per element
sets.sz = (sets.edof * (sets.edof + 1) )/2;	% Number of NNZ values for each Ke using simmetry
sets.tdofs = sets.nnod * sets.dxn;         	% Number of total DOFs in the mesh

%% Creation of global stiffness matrix on CPU (serial)
tic;
K_f = StiffMa_vs(Mesh, MP, sets);	% Assembly on CPU: K
time = toc;
fprintf('\nElapsed time for building K on serial CPU: %f\n',time);

%% Creation of global stiffness matrix on CPU (serial) taking advantage of symmetry
tic;
K_s = StiffMa_vss(Mesh, MP, sets);  % Assembly on CPU (tril(K))
times = toc;
fprintf('Elapsed time for building tril(K) on serial CPU: %f\n',times);

