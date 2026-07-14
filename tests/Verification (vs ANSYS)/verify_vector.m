%% Verify the VECTOR implementation
% 
% This script is used to compare the results between ANSYS and MATLAB for the VECTOT problem.
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.
%
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Institución Universitaria Pascual Bravo, Medellin-Colombia
%       Updated: July 10, 2026. Version: 1.1
%       Created:  18/12/2019. Version: 1.0


%% Problem setup
MP.E = 1;          % Elastic modulus [Pa] (homogeneous, linear, isotropic material)
MP.nu = 0.3;       % Poisson ratio []

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
% Data type
dTE = 'uint32';      % Data precision for "elements" ['uint32', 'uint64']
dTN = 'single';      % Data precision for "nodes" ['single' or 'double']
fnE = str2func(dTE); % Function handle to manage "elements" data type
fnN = str2func(dTN); % Function handle to manage "nodes" data type

% Data type conversion
Mesh.elements = fnE(elements);
Mesh.nodes = fnN(nodes);

% Setting variables
[nel, nxe]    = size(Mesh.elements);
[nnodes, dim] = size(Mesh.nodes);
dxn = 3;            % For vector 3 (UX, UY, UZ). For scalar 1 (Temp)
sets.dTE = dTE;     % Data precision for connectivity array
sets.dTN = dTN;     % Data precision for nodal coordinated
sets.nel = nel;     % Number of finite elements
sets.nnodes = nnodes;  % Number of nodes
sets.nxe = nxe;     % Number of nodes per element
sets.dim = dim;     % Dimension (only 3D for now)
sets.dxn = dxn;     % Number of DOFs per node
sets.edof= dxn*nxe; % Number of DOFs per element
sets.tdofs = dxn*nnodes; % Number of total DOFs in the mesh
sets.sz  = sets.edof * (sets.edof + 1) / 2; % Number of symmetry entries

%% Element stiffness matrix generation: ANSYS vs MATLAB
e = 1; folder = 'ANSYS_vrst2/'; name = [folder,'KE',num2str(e),'.dat'];     % Element to be compared
Ke_ansys = mm_to_msm(name);                                                 % Element stiffness matrix storage
Ke_matlab= eStiff_vs(Mesh.nodes(Mesh.elements(e,:),:), DMatrix(MP.E, MP.nu, sets.dTN), sets.dTN);	% Element stiffnes matrix compute & storage

% tril(ke)
Ke_ansys = tril(Ke_ansys);
Ke_matlab = tril(Ke_matlab);

%%  Global Stiffness matrix generation on CPU: ANSYS vs MATLAB

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

%% Comparison

% ANSYS vs MATLAB (ke)
fig1 = figure('color','none');
ax1 = axes('Parent',fig1,'Color','none');
hold on;
spy3(Ke_ansys,'ob'); 
spy3(Ke_matlab,'.r');
legend(ax1, 'ANSYS','MATLAB'); 
title(ax1,"k_e comparison: ANSYS vs MATLAB. Diff. = " + num2str(norm(Ke_ansys(:)-Ke_matlab(:))))
hold off; 

% ANSYS vs ANSYS (MMF vs HB)
fig2 = figure('color','none');
ax2 = axes('Parent',fig2,'Color','none');
K_afs = tril(K_af);
hold on;
spy(K_afs,'ob');  
spy(K_av,'.r'); 
legend(ax2, 'ANSYS MM format','ANSYS HB format'); 
title(ax2,"tril(K) comparison on ANSYS: MM vs HB formats. Diff. = " + norm(K_av(:)-K_afs(:)) )
hold off; 

% ANSYS vs MATLAB: K
fig3 = figure('color','none');
ax3 = axes('Parent',fig3,'Color','none');
hold on;
spy(K_af2m,'ob');  
spy(K_hf2,'.r'); 
legend(ax3, 'ANSYS','MATLAB'); 
title(ax3,"K comparison: ANSYS vs MATLAB. Diff. = " + norm(K_af2m(:)-K_hf2(:)) )
hold off;    

% ANSYS vs MATLAB (Ks vs tril(K))
fig4 = figure('color','none');
ax4 = axes('Parent',fig4,'Color','none');
K_afs = tril(K_hf);
hold on;
spy(K_afs,'ob');  
spy(K_hs,'.r');      
legend(ax4, 'ANSYS','MATLAB'); 
title(ax4,"tril(K) comparison: ANSYS vs MATLAB. Diff. = " + norm(K_hs(:)-K_afs(:)) )
hold off;
