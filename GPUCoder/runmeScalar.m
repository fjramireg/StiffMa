
% run the whole assembly code on the CPU
addpath('../Common');
addpath('../Utils');

%% Problem setup
c = 1.0;            % Conductivity (homogeneous, linear, isotropic material)
nelx = 50;          % Number of elements on X-direction
nely = 10;          % Number of elements on Y-direction
nelz = 10 ;         % Number of elements on Z-direction
dTypeE = 'uint32';  % Data precision for "elements" ['int32', 'uint32', 'int64', 'uint64' or 'double']
dTypeN = 'double';  % Data precision for "nodes" ['single' or 'double']

%% Mesh generation
[elements, nodes] = CreateMesh(nelx,nely,nelz,dTypeE,dTypeN);

%% Creation of global stiffness matrix
tic;
[iK, jK, Ke] = StiffMas(elements,nodes,c);      % using serial CPU
K = accumarray([iK(:),jK(:)],Ke(:),[],[],[],1); % Assembly of the global stiffness matrix
time = toc;
fprintf('\nTime spend to build K on serial CPU: %f\n',time);

tic;
[iK2, jK2, Ke2] = StiffMas_mex(elements,nodes,c);  % using GPU Coder 
K2 = accumarray([iK2(:),jK2(:)],Ke2(:),[],[],[],1);% Assembly of the global stiffness matrix
times = toc;
fprintf('Time spend to build K with GPU Coder: %f\n',times);

% Difference between results
fprintf('Difference between results: %e\n\n',norm(K(:)-K2(:)));
