%% Load data
% Post_runtime
% load('AssemblyRuntime.mat');


%% Plot figure
figure1 = figure('color',[1,1,1]);
axes1 = axes('Parent',figure1);

pt = plot(nel_CPU_Scalar.^3, FTab_CPU_Scalar.Mean,'-+k',... % CPU_Sca
    nel_GPU_Scalar.^3, FTab_GPU_Scalar.Mean,'-ob',...       % GPU_Sca
    nel_CPU_Vector.^3, FTab_CPU_Vector.Mean,'--*k',...      % CPU_Vec
    nel_GPU_Vector.^3, FTab_GPU_Vector.Mean,'--sb');        % GPU_Vec
   
% LineWidth
nl = size(pt,1);
for i=1:nl
    pt(i).LineWidth = 2;
end

% Labels
xlabel('Number of finite elements');
ylabel('Runtime (s)');

% Create legend
% legend('CS4','GS4',...
%        'CS8','GS8');
legend('CPU-Scalar','GPU-Scalar',...
       'CPU-Vector','GPU-Vector');
legend1 = legend(axes1,'show');
set(legend1, 'NumColumns',1,'Location','northwest');

% Set the remaining axes properties
box(axes1,'on');
% ylim(axes1,[0.001 100000]);
set(axes1,'XGrid','on','XMinorTick','on','XScale','log',...
    'YGrid','on','YMinorTick','on','YScale','log');

