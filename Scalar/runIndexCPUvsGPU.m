
% Script to run the INDEX code on the CPU and GPU, and compare them
addpath('../Common');
addpath('../Utils');

%% Problem setup
c = 1.0;            % Conductivity (homogeneous, linear, isotropic material)
nelx = 100;         % Number of elements on X-direction
nely = 100;         % Number of elements on Y-direction
nelz = 100;         % Number of elements on Z-direction
dTypeE = 'uint32';  % Data precision for "elements" ['uint32', 'uint64' or 'double']

%% Mesh generation
[elements, ~] = CreateMesh(nelx,nely,nelz,dTypeE,'double');

%% Index computation on CPU (symmetry)
tic;
[iKh, jKh] = IndexScalarsas(elements,dTypeE);      % Row/column indices of tril(K)
time_h = toc;
fprintf('Elapsed time for computing row/column indices of tril(K) on serial CPU: %f\n',time_h);

%% Index computation on GPU (symmetry)
d = gpuDevice;
elementsGPU = gpuArray(elements');          % Transfer transposed array to GPU memory
tbs = 256;                                  % Thread Block Size
tic;
[iKd, jKd] = IndexScalarsap(elementsGPU, dTypeE, tbs); % Row/column indices of tril(K)
wait(d);
time_d = toc;
fprintf('Elapsed time for computing row/column indices of tril(K) on parallel GPU: %f\n',time_d);
fprintf('GPU speedup: %f\n',time_h/time_d);

%% Difference between results
fprintf('Difference between results:\n');
fprintf('\tCPU vs GPU (iKh vs iKd): %e\n',norm(double(iKh(:))-double(iKd(:))));
fprintf('\tCPU vs GPU (jKh vs jKd): %e\n',norm(double(jKh(:))-double(jKd(:))));
