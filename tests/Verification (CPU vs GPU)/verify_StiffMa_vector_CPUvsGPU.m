% This script is used to compare the results between CPU and GPU in MATLAB for the VECTOR problem.
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.
%
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Institución Universitaria Pascual Bravo, Medellin-Colombia
%       Created: July 10, 2026. Version: 1.0

%% Problem setup
E = 200e9;          % Elastic modulus [Pa] (homogeneous, linear, isotropic material)
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

%%  Stiffness matrix generation on CPU

% MATLAB Computation on serial CPU
K_h = StiffMa_vss(Mesh, MP, sets);          % MATLAB assembly on CPU: tril(K)

%%  Stiffness matrix generation on GPU
v = ver;
if any(strcmp({v.Name}, 'Parallel Computing Toolbox'))

    % GPU Settings
    d = gpuDevice;
    sets.tbs      = d.MaxThreadsPerBlock;   % Max. Thread Block Size
    sets.numSMs   = d.MultiprocessorCount;  % Number of multiprocessors on the device
    sets.WarpSize = d.SIMDWidth;            % The warp size in threads

    % Memmory transfer
    elementsGPU = gpuArray(Mesh.elements);
    nodesGPU = gpuArray(Mesh.nodes);

    % MATLAB Computation on parallel GPU (Optimized version)
    K_d = StiffMa_vps_opt(elementsGPU, nodesGPU, MP, sets);
    K_d2 = gather(K_d);
end

%% Comparison

if any(strcmp({v.Name}, 'Parallel Computing Toolbox'))

    if ( size(K_d2) ~= size(K_h) )
        error('Mismatch results between CPU and GPU computations.');
    else

        % K sparsity pattern
        fig1 = figure('color','none');
        ax1 = axes('Parent',fig1,'Color','none');
        hold on;
        spy(K_h,'or');         % K_tril from MATLAB Computation on serial CPU
        spy(K_d2,'.b');         % K_tril from MATLAB Computation on parallel GPU
        title("CPU vs GPU");
        legend('Serial CPU','Parallel GPU');
        hold off;

        % K comparison
        Diff_CPUvsGPU_vec = full( K_h(:) - K_d2(:) );
        Diff_CPUvsGPU_esc = norm(Diff_CPUvsGPU_vec);
        fig2 = figure('color','none','InvertHardcopy','off');   % figure background = transparent
        ax2 = axes('Parent',fig2,'Color','none');               % axes background = transparent
        hold on
        plot(ax2, 1:length(Diff_CPUvsGPU_vec), Diff_CPUvsGPU_vec,'b','DisplayName', sprintf('CPU vs GPU. L_2-norm = %d', Diff_CPUvsGPU_esc ) );%,...
        legend(ax2,'show',"Location","best")
        title(ax2, "Global Stiffness Matrix Comparison");
        xlabel(ax2, 'Entry Index');
        ylabel(ax2, 'Difference');
        grid(ax2, 'off');
        box on;
        hold off;

        % K*u comparison
        n = 1:sets.tdofs;
        u_sol = rand(sets.tdofs, 1, dTN);
        f_sol_h = K_h*u_sol;
        f_sol_d = K_d2*u_sol;
        fig3 = figure('color','none','InvertHardcopy','off');   % figure background = transparent
        ax3 = axes('Parent',fig3,'Color','none');               % axes background = transparent
        plot(n, f_sol_h, 'or', n, f_sol_d, '.b', n, f_sol_d-f_sol_h, '--y')
        legend(ax3,{'CPU','GPU', 'Diff.'},"Location","best")
        title(ax3, "Global Stiffness Matrix Comparison: f = K \times u");
        xlabel(ax3, 'Entry Index');
        ylabel(ax3, 'Difference');

        % K\u comparison
        f_sol_h2 = K_h\u_sol;
        f_sol_d2 = K_d2\u_sol;
        fig4 = figure('color','none','InvertHardcopy','off');   % figure background = transparent
        ax4 = axes('Parent',fig4,'Color','none');               % axes background = transparent
        plot(n, f_sol_h2, 'or', n, f_sol_d2, '.b', n, f_sol_d2-f_sol_h2, '--y')
        legend(ax4,{'CPU','GPU', 'Diff.'},"Location","best")
        title(ax4, "Global Stiffness Matrix Comparison: f = K \ u");
        xlabel(ax4, 'Entry Index');
        ylabel(ax4, 'Difference');

    end

end
