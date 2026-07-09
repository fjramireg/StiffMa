% This script is used to compare the results between CPU and GPU in MATLAB for the scalar problem.
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.
%
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Institución Universitaria Pascual Bravo, Medellin-Colombia
%       Created: April 14, 2026. Version: 1.0

%% Problem setup
c = 384.1;          % Conductivity (homogeneous, linear, isotropic material) (W/(m*K))

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
dTN = 'single';     % Data precision for "nodes" ['single' or 'double']
Mesh.nodes = single(nodes);
Mesh.elements = uint32(elements);
[nel, nxe] = size(Mesh.elements);
dxn = 1;            % For vector 3 (UX, UY, UZ). For scalar 1 (Temp)
sets.dTE = dTE;     % Data precision for computing
sets.dTN = dTN;     % Data precision for computing
sets.nel = nel;     % Number of finite elements
sets.nxe = nxe;     % Number of nodes per element
sets.dxn = dxn;     % Number of DOFs per node
sets.edof= dxn*nxe; % Number of DOFs per element
sets.tdofs = dxn*size(nodes,1); % Number of total DOFs in the mesh
sets.sz  = sets.edof * (sets.edof + 1) / 2; % Number of symmetry entries

%%  Stiffness matrix generation on CPU

% MATLAB Computation on serial CPU
K_hs = StiffMa_sss(Mesh, c, sets);                  % MATLAB assembly on CPU: tril(K)

%%  Stiffness matrix generation on GPU
v = ver;
if any(strcmp({v.Name}, 'Parallel Computing Toolbox'))

    % GPU Settings
    d = gpuDevice;
    sets.tbs      = d.MaxThreadsPerBlock;   % Max. Thread Block Size
    sets.numSMs   = d.MultiprocessorCount;  % Number of multiprocessors on the device
    sets.WarpSize = d.SIMDWidth;            % The warp size in threads

    % MATLAB Computation on parallel GPU
    elementsGPU = gpuArray(Mesh.elements);
    nodesGPU = gpuArray(Mesh.nodes);
    K_ds = StiffMa_sps(elementsGPU, nodesGPU', c, sets); % MATLAB stiffness matrix on GPU (tril(K))
    K_ds2 = gather(K_ds);

    % MATLAB Computation on parallel GPU (Optimized version)
    K = StiffMa_sps_opt(elementsGPU, nodesGPU, c, sets);

end

%% Comparison

% figure('color',[1,1,1]);
% spy3(K_hs,'or'); hold on;
% spy3(K_ds2,'.b'); legend('CPU','GPU'); hold off;
% fprintf("MATLAB. CPU vs GPU. Difference: %u\n",norm(K_hs(:)-K_ds2(:)));

% %% Comparison: GPU
% if any(strcmp({v.Name}, 'Parallel Computing Toolbox'))
%     fig3 = figure('color','none','InvertHardcopy','off');   % figure background = transparent
%     ax3 = axes('Parent',fig3,'Color','none');               % axes background = transparent
%     hold on;
%     spy3(K_hs,'or');                                    % K_tril from MATLAB on CPU
%     spy3(gather(K_ds),'.b');                                    % K_tril from MATLAB on GPU
%     legend({'CPU','GPU'},"Location","northeast"); hold off;
%     fprintf("MATLAB. CPU vs GPU. Difference: %u\n",norm(K_hs(:)-K_ds(:)));
%
%     % fig4 = figure('color','none','InvertHardcopy','off');   % figure background = transparent
%     % ax4 = axes('Parent',fig4,'Color','none');               % axes background = transparent
%     % hold on;
%     % spy3(gather(K_ds),'or');                                    % K_tril from MATLAB on CPU
%     % spy3(gather(K_ds2),'.b');                                    % K_tril from MATLAB on GPU
%     % legend({'CPU','GPU'},"Location","northeast"); hold off;
%     % fprintf("MATLAB. CPU vs GPU. Difference: %u\n",norm(K_ds(:)-K_ds2(:)));
% end
