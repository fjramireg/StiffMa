%% Load CPU data & concatenate
load('EStiffPerfTest_WIN_CPU.mat');
FTab_CPU = fullTable;
load('EStiffPerfTest_WIN.mat');
FTab_CPU1 = vertcat(FTab_CPU,fullTable);
clear fullTable;

%% Organize CPU data by category

% Scalar
FTab_CPU1_Scalar = FTab_CPU1(2:4:end,:);
FTab_CPU1_Scalar_single = FTab_CPU1_Scalar(1:2:end,:);
nel_CPU_Scalar_single = [10, 20, 40, 80, 160, 320];
FTab_CPU1_Scalar_double = FTab_CPU1_Scalar(2:2:end,:);
nel_CPU_Scalar_double = [10, 20, 40, 80, 160, 320];

% Vector
FTab_CPU1_Vector = FTab_CPU1(4:4:end,:);
FTab_CPU1_Vector_single = FTab_CPU1_Vector(1:2:end,:);
nel_CPU_Vector_single = [10, 20, 40, 80, 160, 320];
FTab_CPU1_Vector_double = FTab_CPU1_Vector(2:2:end,:);
nel_CPU_Vector_double = [10, 20, 40, 80, 160, 320];


%% Load GPU data & concatenate
load('ETestLNX_GVd32_320.mat', 'fullTable');
FTab_GPU = fullTable;
load('ETestLNX_GSs32_390.mat', 'fullTable');
FTab_GPU = vertcat(FTab_GPU,fullTable);
load('ETestLNX_GVs32_192.mat', 'fullTable');
FTab_GPU = vertcat(FTab_GPU,fullTable(3,:));
load('ETestLNX_GSd32_362.mat', 'fullTable');
FTab_GPU = vertcat(FTab_GPU,fullTable(end,:));
load('ETestLNX_GVd32_188.mat', 'fullTable');
FTab_GPU = vertcat(FTab_GPU,fullTable(end,:));
clear fullTable;

%% Organize GPU data by category

% Scalar
FTab_GPU_Scalar = FTab_GPU(1:2:end,:);
FTab_GPU_Scalar_single = FTab_GPU_Scalar(1:2:end,:);
nel_GPU_Scalar_single = [10, 20, 40, 80, 160, 320, 390];
FTab_GPU_Scalar_double = FTab_GPU_Scalar(2:2:end,:);
nel_GPU_Scalar_double = [10, 20, 40, 80, 160, 320, 362];

% Vector
FTab_GPU_Vector = FTab_GPU(2:2:end,:);
FTab_GPU_Vector_single = FTab_GPU_Vector(1:2:end,:);
nel_GPU_Vector_single = [10, 20, 40, 80, 160, 320, 192];
FTab_GPU_Vector_double = FTab_GPU_Vector(2:2:end,:);
nel_GPU_Vector_double = [10, 20, 40, 80, 160, 320, 188];
