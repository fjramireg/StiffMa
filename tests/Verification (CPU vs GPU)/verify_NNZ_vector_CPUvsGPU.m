% This script is used to compare the results between CPU and GPU in MATLAB for the VECTOR problem.
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.
%
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Institución Universitaria Pascual Bravo, Medellin-Colombia
%       Created: July 10, 2026. Version: 1.0

%% Problem setup
E = 1;          % Elastic modulus [Pa] (homogeneous, linear, isotropic material)
nu = 0.3;       % Poisson ratio []

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
MP.E = fnN(E);
MP.nu = fnN(nu);

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

%%  KE generation on CPU

% MATLAB Computation on serial CPU
Ke_h = eStiff_vssa(Mesh, MP, sets); % Element stiffness matrix computation - All tril(Ke)

%%  KE generation on GPU
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

    % Element matrix computation. Old version
    % Ke_d = eStiff_vpsa(elementsGPU', nodesGPU', MP, sets);   % tril(Ke) for all elements

    % Element matrix computation. Optimized version
    Ke_dopt = eStiff_vpsa_opt(elementsGPU, nodesGPU, MP, sets);    % tril(Ke) for all elements
    % Re-shape to obtain a column vector of size sz-by-nel with contiguos
    % entries for each element (only for proper comparison, not necessary
    % for the assembly as index i,j has the same order)
    Ke_dopt2 = reshape( reshape(Ke_dopt.',sets.nel,sets.sz).', [], 1); %

end

%% Comparison
if any(strcmp({v.Name}, 'Parallel Computing Toolbox'))

    % Values
    % Diff_CPUvsGPU_vec = Ke_h - Ke_d;
    % Diff_CPUvsGPU_esc = norm(Diff_CPUvsGPU_vec);
    Diff_CPUvsGPUopt_vec = Ke_h - Ke_dopt2;
    Diff_CPUvsGPUopt_esc = norm(Diff_CPUvsGPUopt_vec);
    % Diff_GPUvsGPUopt_vec = Ke_d - Ke_dopt2;
    % Diff_GPUvsGPUopt_esc = norm(Diff_GPUvsGPUopt_vec);

    % Show as tex
    % fprintf("Ke: CPU vs GPU. Difference: %u\n",Diff_CPUvsGPU_esc);
    fprintf("Ke: CPU vs GPU optimized. Difference: %u\n",Diff_CPUvsGPUopt_esc);
    % fprintf("Ke: GPU vs GPU optimized. Difference: %u\n",Diff_GPUvsGPUopt_esc);

    % Show as figure
    fig = figure('color','none','InvertHardcopy','off');   % figure background = transparent
    ax = axes('Parent',fig,'Color','none');               % axes background = transparent
    nent = 1:length(Ke_h);
    hold on
    % plot(ax, nent, Diff_CPUvsGPU_vec,'b','DisplayName', sprintf('CPU vs GPU. L_2-norm = %d', Diff_CPUvsGPU_esc ) );%,...
    plot(ax, nent, Diff_CPUvsGPUopt_vec,'b','DisplayName', sprintf('CPU vs GPU opt. L_2-norm = %d', Diff_CPUvsGPUopt_esc ) );%,...
    % plot(ax, nent, Diff_GPUvsGPUopt_vec,'--r','DisplayName', sprintf('GPU vs GPU opt. L_2-norm = %d', Diff_GPUvsGPUopt_esc ) );%,...
    legend(ax,'show',"Location","best")
    title(ax, "Elemental Stiffness Matrix Comparison");
    xlabel(ax, 'Entry Index');
    ylabel(ax, 'Difference');
    grid(ax, 'off');
    box on;
    hold off;
    % saveas(fig, 'stiffness_matrix_comparison.png'); % Save the figure as a PNG file

end
