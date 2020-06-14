figure1 = figure('color',[1,1,1]);
axes1 = axes('Parent',figure1);

pt = plot(nel_Sca_uint32.^3, TabCPU2_Sca_uint32.Mean,'-+k',...   % CPU_Sca_uint32
    nel_Sca_uint32.^3, TabGPU_Sca_uint32.Mean,'-db',...          % GPU_Sca_uint32
    nel_Sca_uint64.^3, TabCPU2_Sca_uint64.Mean,'-^r',...         % CPU_Sca_uint64
    nel_Sca_uint64.^3, TabGPU_Sca_uint64.Mean,'-oy',...          % GPU_Sca_uint64
    nel_Vec_uint32.^3, TabCPU2_Vec_uint32.Mean,'--.k',...        % CPU_Vec_uint32
    nel_Vec_uint32.^3, TabGPU_Vec_uint32.Mean,'--sb',...         % GPU_Vec_uint32
    nel_Vec_uint64.^3, TabCPU2_Vec_uint64.Mean,'--xr',...        % CPU_Vec_uint64
    nel_Vec_uint64.^3, TabGPU_Vec_uint64.Mean,'--*y');           % GPU_Sca_uint64

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
legend('CS32','GS32',...
       'CS64','GS64',...
       'CV32','GV32',...
       'CV64','GV64');
legend1 = legend(axes1,'show');
set(legend1, 'NumColumns',2,'Location','best');

% Set the remaining axes properties
box(axes1,'on');
set(axes1,'XGrid','on','XMinorTick','on','XScale','log',...
    'YGrid','on','YMinorTick','on','YScale','log');