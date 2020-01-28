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
E = 200e9;          % Elastic modulus [Pa] (homogeneous, linear, isotropic material)
nu = 0.3;           % Poisson ratio []

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

%%  Stiffness matrix generation

% ANSYS Computation
% StiffMavansys_mac(elements,nodes,E,nu);                 % ANSYS macro to generate tril(K)
% !ansys171 -b -i StiffMavansys.mac -o StiffMavansys.out  % Execute ANSYS (must be on systeme path)
% Copy generated files to the folder "ANSYS_vrst"

% ANSYS importation
K_af = mm_to_msm ('ANSYS_vrst/STIFF_ANSYS.mmf');        % Import ANSYS result: K
K_av = hb_to_msm ('ANSYS_vrst/STIFF_ANSYS.hb');         % Import ANSYS result: tril(K)
figure,spy(K_af,'ob'); hold on; spy(K_av,'.r'); hold off; % Graphical comparison
K_afs = tril(K_af);
fprintf("\tANSYS vs ANSYS (MMF vs HB). Difference: %u\n",norm(K_av(:)-K_afs(:)));

% ANSYS import Ke and MATLAB compute K
K_af2 = StiffMavansys_import('ANSYS_vrst/');             % Import ANSYS result: K (built from element matrices)
[~, MapVec, DOF] = importMappingFile('ANSYS_vrst/STIFF_ANSYS.mapping'); % Import the ANSYS reorder vector
UX_dofs = DOF == 'UX'; MapVec(UX_dofs) = 3*MapVec(UX_dofs) - 2;
UY_dofs = DOF == 'UY'; MapVec(UY_dofs) = 3*MapVec(UY_dofs) - 1;
UZ_dofs = DOF == 'UZ'; MapVec(UZ_dofs) = 3*MapVec(UZ_dofs) - 0;
K_af2m = K_af2(MapVec, MapVec);
figure,spy(K_af,'ob'); hold on; spy(K_af2m,'.r'); hold off; % Graphical comparison
fprintf("\tANSYS compute vs MATLAB assembly. Difference: %u\n",norm(K_af2m(:)-K_af(:)));

% MATLAB Computation on serial CPU
K_hf = StiffMav(elements,nodes,E,nu);                   % MATLAB assembly on CPU: K
K_hf2= K_hf(MapVec,MapVec);                             % Reordered K in MATLAB with ANSYS permutation vector
figure,spy(K_af,'ob'); hold on; spy(K_hf2,'.r'); hold off; % Graphical comparison
fprintf("\tANSYS vs MATLAB. Difference: %u\n", norm(K_hf2(:)-K_af(:)));

% MATLAB Computation on serial CPU (symmetry)
% K_hs = StiffMass(elements,nodes,c);                 % MATLAB assembly on CPU: tril(K)
% % K_hs2= tril((K_hs(MapVec,MapVec) + K_hs(MapVec,MapVec)')/2);                         % Reorder K in MATLAB as ANSYS result
%
% % Graphical comparison
% figure('color',[1,1,1]); spy(K_af,'or'); hold on;   % ANSYS vs MATLAB: 2D
% spy(K_hf2,'.b'); legend('ANSYS','MATLAB'); hold off;
%
% figure('color',[1,1,1]); spy3(K_af,'or'); hold on;  % ANSYS vs MATLAB: 3D
% spy3(K_hf2,'.b'); legend('ANSYS','MATLAB'); hold off;
% fprintf("ANSYS vs MATLAB. Difference: %u\n",norm(K_af(:)-K_hf2(:)));
