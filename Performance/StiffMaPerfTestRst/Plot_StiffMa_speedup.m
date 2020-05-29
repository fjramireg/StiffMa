% function Plot_speedup

% Load data
% Post_data_stiffma
% load('StiffMaRuntime.mat');

%% Speedup
Speedup_scalar = FTab_CPU_Scalar.Mean ./ FTab_GPU_Scalar.Mean(1:5);
Speedup_vector = FTab_CPU_Vector.Mean ./ FTab_GPU_Vector.Mean(1:5);
nel = [10, 20, 40, 80, 160];


%% Graph
figure1 = figure('color',[1,1,1]);
axes1 = axes('Parent',figure1);

pt = plot(nel.^3, Speedup_scalar,'-+k',...   % CPU/GPU_Scalar
    nel.^3, Speedup_vector,'--ob');          % CPU/GPU_Vector

% LineWidth
nl = size(pt,1);
for i=1:nl
    pt(i).LineWidth = 2;
end

% Labels
xlabel('Number of finite elements');
ylabel('Speedup ratio');

% Create legend{Fig:IndexSpeedup}
legend('Scalar', 'Vector');
legend1 = legend(axes1,'show');
set(legend1, 'Location','best'); % 'NumColumns',2,

% Set the remaining axes properties
box(axes1,'on');
set(axes1,'XGrid','on','XMinorTick','on','XScale','log',...
    'YGrid','on','YMinorTick','on','YScale','log');
