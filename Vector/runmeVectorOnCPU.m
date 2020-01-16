% This script is used to generate the global stiffness matrix K for the VECTOT
% problem (linear static elasticiy). 
% 
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.
% 
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  18/12/2019. Version: 1.0

%% run the whole assembly code on the CPU
addpath('../Common');
addpath('../Utils');

%% Mesh generation
nelx = 10;           % Number of elements on X-direction
nely = 10;           % Number of elements on Y-direction
nelz = 10;           % Number of elements on Z-direction
dTypeE = 'uint32';  % Data precision for "elements" ['int32', 'uint32', 'int64', 'uint64' or 'double']
dTypeN = 'double';  % Data precision for "nodes" ['single' or 'double']
[elements, nodes] = CreateMesh(nelx,nely,nelz,dTypeE,dTypeN);

%% Material properties
E = 200e9;          % Elastic modulus (homogeneous, linear, isotropic material)
nu = 0.3;           % Poisson ratio

%% Creation of global stiffness matrix on CPU (serial)
tic;
K_f = StiffMav(elements,nodes,E,nu);               % Assembly on CPU: K
time = toc;
fprintf('\nTime spend to build K on serial CPU: %f\n',time);

% K_s = StiffMatGenVcSymCPU(elements,nodes,E,nu);         % Assembly on CPU (symmetric)
%% Creation of global stiffness matrix on CPU (serial)
K_h = StiffMatGenVc(elements,nodes,E,nu);               % Assembly on CPU
K_s = StiffMatGenVcSymCPU(elements,nodes,E,nu);         % Assembly on CPU (symmetric)
