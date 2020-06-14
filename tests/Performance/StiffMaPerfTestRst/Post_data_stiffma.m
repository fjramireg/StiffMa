%% Load CPU data & concatenate
load('StiffMaWIN_CVdu32_160.mat', 'fullTable');
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
load('StiffMaLNX_GVdu32_160.mat', 'fullTable');
FTab_GPU = fullTable;
load('StiffMaLNX_GSdu32_195.mat', 'fullTable'); % Scalar largest
FTab_GPU = vertcat(FTab_GPU,fullTable);
load('StiffMaLNX_GVdu32_95.mat', 'fullTable');
FTab_GPU = vertcat(FTab_GPU,fullTable);
clear fullTable;

% Organize GPU data by category
% Scalar
FTab_GPU_Scalar = FTab_GPU(1:2:end,:);
nel_GPU_Scalar = [10, 20, 40, 80, 160, 195];
% Vector
FTab_GPU_Vector = FTab_GPU(2:2:end,:);
nel_GPU_Vector = [10, 20, 40, 80, 160, 95];

%% Save data
save StiffMaRuntime.mat
