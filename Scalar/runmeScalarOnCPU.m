
% run the whole assembly code on the CPU
addpath('../Common');
addpath('../Utils');

%% Problem setup
c = 1.0;            % Conductivity (homogeneous, linear, isotropic material)
nelx = 10;          % Number of elements on X-direction
nely = 10;          % Number of elements on Y-direction
nelz = 10 ;         % Number of elements on Z-direction
dTypeE = 'uint32';  % Data precision for "elements" ['int32', 'uint32', 'int64', 'uint64' or 'double']
dTypeN = 'double';  % Data precision for "nodes" ['single' or 'double']
PlotE = 0;          % Plot the elements and their numbers (1 to plot)
PlotN = 0;          % Plot the nodes and their numbers (1 to plot)

%% Mesh generation
[elements, nodes] = CreateMesh(nelx,nely,nelz,dTypeE,dTypeN,PlotE,PlotN);

%% Creation of global stiffness matrix on CPU (serial)
tic;
K = StiffMas(elements,nodes,c);     % Assembly on CPU
time = toc;
fprintf('\nTime spend to build K on serial CPU: %f\n',time);

% TODO: Takes longer time
% FIXME: Only works for variables defined at compiled time
% tic;
% K_d = StiffMas_mex(elements,nodes,c);     % Assembly on GPU
% time_d = toc;
% fprintf('Time spend to build K on parallel GPU: %f\n',time_d);

% ______________________________________________________________________________
% V2
% ______________________________________________________________________________

% tic;
% K2 = StiffMas2(elements,nodes,c);     % Assembly on CPU
% time2 = toc;
% fprintf('Time spend to build K on serial CPU (v2): %f\n',time2);

% TODO: Takes longer time
% FIXED the for the variable mesh
% tic;
% K2_d = StiffMas2_mex(elements,nodes,c);     % Assembly on GPU
% time2_d = toc;
% fprintf('Time spend to build K on parallel GPU (v2): %f\n',time2_d);

% ______________________________________________________________________________
% V3
% ______________________________________________________________________________

% tic;
% [iK, jK, Ke] = StiffMas3(elements,nodes,c);     % Assembly on CPU
% K3 = accumarray([iK(:),jK(:)],Ke(:),[],[],[],1); % Assembly of the global stiffness matrix
% time3 = toc;
% fprintf('Time spend to build K on serial CPU (v3): %f\n',time3);
% 
% % TODO: Takes longer time. Partial fix: Adding the compiler flags '--fmad=false' and '-arch=sm_50'
% % FIXME: Leave the variable on the GPU
% tic;
% [iK_d, jK_d, Ke_d] = StiffMas3_mex(elements,nodes,c);     % Assembly on GPU
% K3_d = accumarray([iK_d(:),jK_d(:)],Ke_d(:),[],[],[],1); % Assembly of the global stiffness matrix
% time3_d = toc;
% fprintf('Time spend to build K on parallel GPU (v3): %f\n',time3_d);

% ______________________________________________________________________________
% V4
% ______________________________________________________________________________

% % Same as V3, but it is necessary to create 'StiffMas4.prj'
% tic;
% [iK, jK, Ke] = StiffMas4(elements,nodes,c);     % Assembly on CPU
% K4 = accumarray([iK(:),jK(:)],Ke(:),[],[],[],1); % Assembly of the global stiffness matrix
% time4 = toc;
% fprintf('Time spend to build K on serial CPU (v4): %f\n',time4);
% 
% tic;
% % FIXME: Outputs are gpuArray, but inputs can't. Variable size inputs not supported
% % elements_d = gpuArray(elements);
% % nodes_d = gpuArray(nodes);
% % c_d = gpuArray(c);
% [iK_d, jK_d, Ke_d] = StiffMas4_mex(elements,nodes,c);     % Assembly on GPU
% K4_d = accumarray([iK_d(:),jK_d(:)],Ke_d(:),[],[],[],1); % Assembly of the global stiffness matrix
% time4_d = toc;
% fprintf('Time spend to build K on parallel GPU (v4): %f\n',time4_d);

% ______________________________________________________________________________
% V5: Source code
% ______________________________________________________________________________

% Same as V3, but it is necessary to create 'StiffMas5.prj'
tic;
K5 = StiffMas5(elements,nodes,c);     % Assembly on CPU
time5 = toc;
fprintf('Time spend to build K on serial CPU (v5): %f\n',time5);

tic;
K5_d = StiffMas5_mex(elements,nodes,c);     % Assembly on GPU
time5_d = toc;
fprintf('Time spend to build K on parallel GPU (v5): %f\n',time5_d);

% ______________________________________________________________________________
% Results verification
% ______________________________________________________________________________
% figure; spy(K3)
% figure; spy(K)
% norm(K3(:)-K(:))

%% Creation of global stiffness matrix on CPU (serial) taking advantage of symmetry
% tic;
% K_s = StiffMass(elements,nodes,c);  % Assembly on CPU (tril(K))
% times = toc;
% fprintf('Time spend to build tril(K) on serial CPU: %f\n',times);
