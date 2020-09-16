%% Load CPU data & concatenate
load('ATestWIN_C_Vd32_160.mat', 'fullTable');
FTab_CPU = fullTable(2:2:end,:);
clear fullTable;

% Organize CPU data by category
% Scalar
FTab_CPU_Scalar = FTab_CPU(1:2:end,:);
nel_CPU_Scalar = [10, 20, 40, 80, 160];
% Vector
FTab_CPU_Vector = FTab_CPU(2:2:end,:);
nel_CPU_Vector = [10, 20, 40, 80, 160];


%% Load GPU data & concatenate
load('ATestLNX_G_Vd32_320.mat', 'fullTable');
FTab_GPU = fullTable;
load('ATestLNX_G_Sd32_195.mat', 'fullTable');
FTab_GPU = vertcat(FTab_GPU, fullTable(end,:));
load('ATestLNX_G_Vd32_96.mat', 'fullTable');
FTab_GPU = vertcat(FTab_GPU, fullTable(end,:));
clear fullTable;

% Organize GPU data by category
% Scalar
FTab_GPU_Scalar = FTab_GPU(1:2:end,:);
nel_GPU_Scalar = [10, 20, 40, 80, 160, 320, 195];
% Vector
FTab_GPU_Vector = FTab_GPU(2:2:end,:);
nel_GPU_Vector = [10, 20, 40, 80, 160, 320, 96];

save AssemblyRuntime.mat
