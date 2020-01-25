
% Script to run the INDEX code on the CPU and GPU, and compare them
addpath('../Common');
addpath('../Utils');

%% Problem setup
c = 1.0;            % Conductivity (homogeneous, linear, isotropic material)
nelx = 10;          % Number of elements on X-direction
nely = 10;          % Number of elements on Y-direction
nelz = 10;          % Number of elements on Z-direction
dTypeE = 'uint32';  % Data precision for "elements" ['uint32', 'uint64']

%% Mesh generation
[elements, ~] = CreateMesh(nelx,nely,nelz,dTypeE,'double');

%% Settings
d = gpuDevice;
settings.dTE      = dTypeE;                 % Data precision for computing
settings.tbs      = 1024;                   % Max. Thread Block Size
settings.nel      = size(elements,1);       % Number of finite elements
settings.numSMs   = d.MultiprocessorCount;  % Number of multiprocessors on the device
settings.WarpSize = d.SIMDWidth;            % The warp size in threads

%% Index computation on CPU (symmetry)
tic;
[iKh, jKh] = IndexScalarsas(elements, settings);	% Row/column indices of tril(K)
time_h = toc;
fprintf('Elapsed time for computing row/column indices of tril(K) on serial CPU: %f\n',time_h);

%% Index computation on GPU (symmetry)
elementsGPU = gpuArray(elements');                  % Transfer transposed array to GPU memory
tic;
[iKd, jKd] = IndexScalarsap(elementsGPU, settings); % Row/column indices of tril(K)
wait(d);
time_d = toc;
fprintf('Elapsed time for computing row/column indices of tril(K) on parallel GPU: %f\n',time_d);
fprintf('GPU speedup: %f\n',time_h/time_d);

%% Difference between results
fprintf('Difference between results:\n');
fprintf('\tCPU vs GPU (iKh vs iKd): %e\n',norm(double(iKh(:))-double(iKd(:))));
fprintf('\tCPU vs GPU (jKh vs jKd): %e\n',norm(double(jKh(:))-double(jKd(:))));
