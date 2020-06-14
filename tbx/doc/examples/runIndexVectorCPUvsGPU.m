% Runs the INDEX vector code. CPU vs GPU
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
nelx = 1000;          % Number of elements on X-direction
nely = 10;          % Number of elements on Y-direction
nelz = 10;          % Number of elements on Z-direction
dTE = 'uint32';     % Data precision for "elements" ['uint32', 'uint64']
dTN = 'single';     % Data precision for "nodes" ['single' or 'double']
[elements, ~] = CreateMesh(nelx,nely,nelz,dTE,dTN);
[nel, nxe] = size(elements);

%% Material properties
MP.E = 200e9;      	% Elastic modulus (homogeneous, linear, isotropic material)
MP.nu = 0.3;       	% Poisson ratio

%% General Settings
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

%% Index computation on CPU
tic;
[iKhf, jKhf] = Index_vsa(elements, sets);	% Row/column indices of tril(K)
times = toc;
fprintf('Elapsed time for computing row/column indices of K on serial CPU: %f\n',times);

%% Index computation on CPU (vectorized)
tic;
[iKhfv, jKhfv] = Index_va(elements', sets);	% Row/column indices of K
time1 = toc;
fprintf('Elapsed time for computing row/column indices of K on CPU (vectorized): %f\n',time1);
fprintf('\tSpeedup (loop vs vectorized): %f\n',times/time1);
fprintf('Difference between results:\n');
fprintf('\tShould be %d!: %d\n', sets.edof^2*sets.nel, sum(iKhf==iKhfv));
fprintf('\tShould be %d!: %d\n\n', sets.edof^2*sets.nel, sum(jKhf==jKhfv));

%% Index computation on CPU (symmetry)
tic;
[iKh, jKh]  = Index_vssa(elements, sets); % Computation of row/column indices of tril(K)
time_h = toc;
fprintf('Elapsed time for computing row/column indices of tril(K) on serial CPU: %f\n',time_h);
fprintf('\tCPU speedup (Full vs Symmetry): %f\n',times/time_h);

%% Index computation on GPU (symmetry)
tic;
elementsGPU = gpuArray(elements');     % Transfer transposed array to GPU memory
[iKd, jKd]  = Index_vpsa(elementsGPU, sets); % Computation of row/column indices of tril(K)
wait(d);
time_d = toc;
fprintf('Elapsed time for computing row/column indices of tril(K) on parallel GPU: %f\n',time_d);
fprintf('\tGPU speedup: %f\n',time_h/time_d);

%% Difference between results
fprintf('Difference between results:\n');
fprintf('\tCPU vs GPU (iKh vs iKd): %e\n',norm(double(iKh(:))-double(iKd(:))));
fprintf('\tCPU vs GPU (jKh vs jKd): %e\n',norm(double(jKh(:))-double(jKd(:))));
