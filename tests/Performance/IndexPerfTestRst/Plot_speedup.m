% function Plot_speedup

% Load data
% load('IndexPerfRst_post2.mat');


figure1 = figure('color',[1,1,1]);
axes1 = axes('Parent',figure1);

pt = plot(nel_Sca_uint32.^3, TabCPU2_Sca_uint32.Mean./TabGPU_Sca_uint32.Mean,'-+k',...   % CPU/GPU_Sca_uint32
    nel_Sca_uint64.^3, TabCPU2_Sca_uint64.Mean./TabGPU_Sca_uint64.Mean,'-ob',...         % CPU/GPU_Sca_uint64
    nel_Vec_uint32.^3, TabCPU2_Vec_uint32.Mean./TabGPU_Vec_uint32.Mean,'--xr',...        % CPU/GPU_Vec_uint32
    nel_Vec_uint64.^3, TabCPU2_Vec_uint64.Mean./TabGPU_Vec_uint64.Mean,'--sy');          % CPU/GPU_Vec_uint64

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
legend('Scalar-uint32',...
       'Scalar-uint64',...
       'Vector-uint32',...
       'Vector-uint64');
legend1 = legend(axes1,'show');
set(legend1, 'Location','best'); % 'NumColumns',2,

% Set the remaining axes properties
box(axes1,'on');
set(axes1,'XGrid','on','XMinorTick','on','XScale','log',...
    'YGrid','on','YMinorTick','on','YScale','log');
