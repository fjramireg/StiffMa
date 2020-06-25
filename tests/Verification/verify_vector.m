% This script is used to compare the results between ANSYS and MATLAB, and
% between CPU and GPU for the VECTOT problem.
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.
%
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  18/12/2019. Version: 1.0

%% verify the SCALAR implementation
addpath('../Common');
addpath('../Utils');
addpath('../Vector');

%% Problem setup
MP.E = 1;          % Elastic modulus [Pa] (homogeneous, linear, isotropic material)
MP.nu = 0.3;           % Poisson ratio []

%% Mesh generation
nodes = [0, 0, 0;   % node 1
    0, 1, 0;        % node 2
    0, 1, 1;        % node 3
    0, 0, 1;        % node 4
    0, 0.5, 0;      % node 5
    0, 1, 0.5;      % node 6
    0, 0.5, 1;      % node 7
    0, 0, 0.5;      % node 8
    0, 0.5, 0.5;    % node 9
    1, 0, 0;        % node 10
    1, 1.5, 0;      % node 11
    1, 1.5, 1.5;    % node 12
    1, 0, 1.5;      % node 13
    1, .75, 0;      % node 14
    1, 1.5, .75;    % node 15
    1, .75, 1.5;    % node 16
    1, 0, .75;      % node 17
    1, 0.75, 0.75;  % node 18
    2, 0, 0;        % node 19
    2, 2, 0;        % node 20
    2, 2, 2;        % node 21
    2, 0, 2;        % node 22
    2, 1, 0;        % node 23
    2, 2, 1;        % node 24
    2, 1, 2;        % node 25
    2, 0, 1;        % node 26
    2, 1, 1;];      % node 27

elements = [1 5 9 8 10 14 18 17;% Element 1
    5 2 6 9 14 11 15 18;        % Element 2
    9 6 3 7 18 15 12 16;        % Element 3
    8 9 7 4 17 18 16 13;        % Element 4
    10 14 18 17 19 23 27 26;    % Element 5
    14 11 15 18 23 20 24 27;    % Element 6
    18 15 12 16 27 24 21 25;    % Element 7
    17 18 16 13 26 27 25 22];   % Element 8

%% Settings
dTE = 'uint32';     % Data precision for "elements" ['uint32', 'uint64']
dTN = 'double';     % Data precision for "nodes" ['single' or 'double']
Mesh.nodes = nodes;
Mesh.elements = uint32(elements);
[nel, nxe] = size(Mesh.elements);
dxn = 3;            % For vector 3 (UX, UY, UZ). For scalar 1 (Temp)
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

%% Element stiffness matrix generation
e = 1;
folder = 'ANSYS_vrst2/';
name = [folder,'KE',num2str(e),'.dat'];
Ke_ansys = mm_to_msm(name);    % Element stiffness matrix storage
Ke_matlab= eStiff_vs(Mesh.nodes(Mesh.elements(e,:),:), DMatrix(MP.E, MP.nu, sets.dTN), sets.dTN);	% Element stiffnes matrix compute & storage

Ke_ansys = tril(Ke_ansys);
Ke_matlab = tril(Ke_matlab);

fprintf("\n\tANSYS vs MATLAB. Difference of ke: %u\n", norm(Ke_ansys(:)-Ke_matlab(:)) );

% figure('color',[1,1,1]);
% spy3(Ke_ansys,'ob'); hold on;
% spy3(Ke_matlab,'.r');
% legend('ANSYS','MATLAB'); hold off; % Graphical comparison 3D

figure('color',[1,1,1]);
spy(Ke_ansys,'ob'); hold on;
spy(Ke_matlab,'.r');
legend('ANSYS','MATLAB'); hold off; % Graphical comparison 2D

%%  Stiffness matrix generation

% ANSYS Computation
% StiffMavansys_mac(elements,nodes,E,nu);                 % ANSYS macro to generate tril(K)
% !ANSYS193 -b -smp -np 1 -i StiffMavansys.mac -o StiffMavansys.out  % Execute ANSYS (must be on systeme path)
% Copy generated files to the folder "ANSYS_vrst"

% ANSYS importation
[~, MapVec, DOF] = importMappingFile('ANSYS_vrst2/STIFF_ANSYS.mapping'); % Import the ANSYS reorder vector
K_af = mm_to_msm ('ANSYS_vrst2/STIFF_ANSYS.mmf');        % Import ANSYS result: K
K_av = hb_to_msm ('ANSYS_vrst2/STIFF_ANSYS.hb');         % Import ANSYS result: tril(K)

% ANSYS import Ke and MATLAB compute K
K_af2 = StiffMavansys_import('ANSYS_vrst2/');             % Import ANSYS result: K (built from element matrices)
UX_dofs = DOF == 'UX'; MapVec(UX_dofs) = 3*MapVec(UX_dofs) - 2;
UY_dofs = DOF == 'UY'; MapVec(UY_dofs) = 3*MapVec(UY_dofs) - 1;
UZ_dofs = DOF == 'UZ'; MapVec(UZ_dofs) = 3*MapVec(UZ_dofs) - 0;
K_af2m = K_af2(MapVec, MapVec);

% MATLAB Computation on serial CPU                          % MATLAB assembly on CPU: K
K_hf = StiffMa_vs(Mesh, MP, sets);                   
K_hf2 = K_hf(MapVec, MapVec);

% MATLAB Computation on serial CPU (symmetry)               % MATLAB assembly on CPU: tril(K)
K_hs = StiffMa_vss(Mesh, MP, sets);                 
% K_hs2= tril((K_hs(MapVec,MapVec) + K_hs(MapVec,MapVec)')/2);  % Reorder K in MATLAB as ANSYS result

% MATLAB Computation on parallel GPU (symmetry)             % MATLAB assembly on GPU: tril(K)
elementsGPU = gpuArray(Mesh.elements');
nodesGPU = gpuArray(Mesh.nodes');
K_ds = StiffMa_vps(elementsGPU, nodesGPU, MP, sets);                 

%% Comparison

% Graphical comparison                                  % ANSYS vs ANSYS (MMF vs HB: 2D
K_afs = tril(K_af);
figure('color',[1,1,1]);
spy(K_afs,'ob'); hold on; spy(K_av,'.r'); hold off; % Graphical comparison
fprintf("\tANSYS vs ANSYS (MMF vs HB). Difference of tril(K): %u\n",norm(K_av(:)-K_afs(:)));

% Graphical comparison                                  % ANSYS vs ANSYS (MATLAB assembly): 2D
figure('color',[1,1,1]);
spy(K_af2m,'ob'); hold on; spy(K_hf2,'.r'); hold off;     % Graphical comparison
fprintf("\tANSYS compute vs MATLAB assembly. Difference: %u\n", norm(K_af2m(:)-K_hf2(:)));

% Graphical comparison                                  % MATLAB vs MATLAB (Ks vs tril(K)): 2D
K_hfs = tril(K_hf);
figure('color',[1,1,1]);
spy(K_hs,'ob'); hold on; spy(K_hfs,'.r'); hold off;     % Graphical comparison
fprintf("\tMATLAB vs MATLAB. Difference of Ks vs tril(K): %u\n", norm(K_hs(:)-K_hfs(:)));

% Graphical comparison                                  % MATLAB vs MATLAB (CPU vs GPU): 2D
figure('color',[1,1,1]);
K_ds2 = gather(K_ds);
spy(K_hs,'ob'); hold on; spy(K_ds2,'.r'); hold off; % Graphical comparison
fprintf("\tMATLAB vs MATLAB. Difference of Ks (CPU vs GPU): %u\n",norm(K_hs(:)-K_ds2(:)));
