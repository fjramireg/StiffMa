% function Plot_speedup

% Load data
load('EStiff_runtime.mat');

Speedup_sca_single = FTab_CPU1_Scalar_single.Mean ./ FTab_GPU_Scalar_single.Mean(1:6);
Speedup_sca_double = FTab_CPU1_Scalar_double.Mean ./ FTab_GPU_Scalar_double.Mean(1:6);
Speedup_vec_single = FTab_CPU1_Vector_single.Mean ./ FTab_GPU_Vector_single.Mean(1:6);
Speedup_vec_double = FTab_CPU1_Vector_double.Mean ./ FTab_GPU_Vector_double.Mean(1:6);
Speedup = [Speedup_sca_single, Speedup_sca_double, Speedup_vec_single, Speedup_vec_double];
nel = [10, 20, 40, 80, 160, 320];


figure1 = figure('color',[1,1,1]);
axes1 = axes('Parent',figure1);

pt = plot(nel.^3, Speedup_sca_single,'-+k',...   % CPU/GPU_Sca_single
    nel.^3, Speedup_sca_double,'-ob',...         % CPU/GPU_Sca_double
    nel.^3, Speedup_vec_single,'--xr',...        % CPU/GPU_Vec_single
    nel.^3, Speedup_vec_double,'--sy');          % CPU/GPU_Vec_double

% LineWidth
nl = size(pt,1);
for i=1:nl
    pt(i).LineWidth = 2;
end

% Marker and color
% pt(1).Marker = '+';
% pt(2).Marker = 'd';
% pt(3).Marker = '*';
pt(4).Color = [0.75, 0.75, 0]; % pt(4).Marker = '^';

% Labels
xlabel('Number of finite elements');
ylabel('Speedup ratio');

% Create legend{Fig:IndexSpeedup}
legend('Scalar-single',...
    'Scalar-double',...
    'Vector-single',...
    'Vector-double');
legend1 = legend(axes1,'show');
set(legend1, 'Location','best'); % 'NumColumns',2,

% Set the remaining axes properties
box(axes1,'on');
set(axes1,'XGrid','on','XMinorTick','on','XScale','log',...
    'YGrid','on','YMinorTick','on','YScale','log');
