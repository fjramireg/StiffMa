%% Load data
% Post_data


%% Plot figure
figure1 = figure('color',[1,1,1]);
axes1 = axes('Parent',figure1);

pt = plot(nel_CPU_Scalar_single.^3, FTab_CPU1_Scalar_single.Mean,'-+k',...   % CPU_Sca_single
    nel_GPU_Scalar_single.^3, FTab_GPU_Scalar_single.Mean,'-db',...          % GPU_Sca_single
    nel_CPU_Scalar_double.^3, FTab_CPU1_Scalar_double.Mean,'-^r',...         % CPU_Sca_double
    nel_GPU_Scalar_double.^3, FTab_GPU_Scalar_double.Mean,'-oy',...          % GPU_Sca_double
    nel_CPU_Vector_single.^3, FTab_CPU1_Vector_single.Mean,'--.k',...        % CPU_Vec_single
    nel_GPU_Vector_single.^3, FTab_GPU_Vector_single.Mean,'--sb',...         % GPU_Vec_single
    nel_CPU_Vector_double.^3, FTab_CPU1_Vector_double.Mean,'--xr',...        % CPU_Vec_double
    nel_GPU_Vector_double.^3, FTab_GPU_Vector_double.Mean,'--*y');           % GPU_Sca_double

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
% pt(5).Marker = 'o';
% pt(6).Marker = '.';
% pt(7).Marker = 's';
pt(8).Color = [0.75, 0.75, 0];% pt(8).Marker = 'x';

% Labels
xlabel('Number of finite elements');
ylabel('Runtime (s)');

% Create legend
% legend('CS4','GS4',...
%        'CS8','GS8',...
%        'CV4','GV4',...
%        'CV8','GV8');
legend('CPU-Scalar-single','GPU-Scalar-single',...
       'CPU-Scalar-double','GPU-Scalar-double',...
       'CPU-Vecor-single','GPU-Vecor-single',...
       'CPU-Vector-double','GPU-Vector-double');
legend1 = legend(axes1,'show');
set(legend1, 'NumColumns',1,'Location','best','FontSize',8);

% Set the remaining axes properties
box(axes1,'on');
% ylim(axes1,[0.001 100000]);
set(axes1,'XGrid','on','XMinorTick','on','XScale','log',...
    'YGrid','on','YMinorTick','on','YScale','log','Ylim',[0.001 100000],...
    'YTick',[0.001 0.01 0.1 1 10 100 1000 10000 100000],'YTickLabel',...
    {'10^{-3}','10^{-2}','10^{-1}','10^{0}','10^{1}','10^{2}','10^{3}','10^{4}','10^{5}'});

