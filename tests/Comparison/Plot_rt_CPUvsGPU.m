% Plot data
% Comparison between optimized CPU code and GPU implementation

%% Load data
load('StiffMa_CPUvsGPU/Comparison-rst.mat');

%% Clasify data
% CPU
tab_index_CPU = fullTable(1:6:end,:);
tab_keall_CPU = fullTable(2:6:end,:);
tab_assem_CPU = fullTable(3:6:end,:);
% GPU
tab_index_GPU = fullTable(4:6:end,:);
tab_keall_GPU = fullTable(5:6:end,:);
tab_assem_GPU = fullTable(6:6:end,:);

%% X-Data
% # of elements
nel = 10:10:100;
x = nel.^3;

%% Y-Data
% Index-CPU
y_iCPU = tab_index_CPU.Mean;
e_iCPU = tab_index_CPU.StandardDeviation;
% Ke_all-CPU
y_kCPU = tab_keall_CPU.Mean;
e_kCPU = tab_keall_CPU.StandardDeviation;
% Asembly-CPU
y_aCPU = tab_assem_CPU.Mean;
e_aCPU = tab_assem_CPU.StandardDeviation;

% Index-GPU
y_iGPU = tab_index_GPU.Mean;
e_iGPU = tab_index_GPU.StandardDeviation;
% Ke_all-GPU
y_kGPU = tab_keall_GPU.Mean;
e_kGPU = tab_keall_GPU.StandardDeviation;
% Asembly-GPU
y_aGPU = tab_assem_GPU.Mean;
e_aGPU = tab_assem_GPU.StandardDeviation;

%% Plot figure
% figure1 = figure('color',[1,1,1]);
% axes1 = axes('Parent',figure1);
% 
% pt = plot(  x, y_iCPU, '-k*', ...
%             x, y_iGPU, '-.k+', ...
%             x, y_kCPU, '-bs', ...
%             x, y_kGPU, '--bo', ...
%             x, y_aCPU, '-rx',...
%             x, y_aGPU, ':r^');
% 
% for i=1:6
%     pt(i).LineWidth = 2;
% end
% 
% % Labels
% xlabel('Number of elements');
% ylabel('Runtime (s)');
% 
% % Create legend
% legend('Indices on CPU', 'Indices on GPU',...
%        'Numerical integration on CPU', 'Numerical integration on GPU',...
%        'Assembly on CPU (fsparse)', 'Assembly on GPU (sparse)');
% legend1 = legend(axes1,'show');
% set(legend1, 'NumColumns',1,'Location','northwest','Interpreter','latex');
% 
% % Set the remaining axes properties
% box(axes1,'on');
% ylim(axes1,[4e-4 11]);
% set(axes1,'XGrid','on','XMinorTick','on','XScale','log',...
%           'YGrid','on','YMinorTick','on','YScale','log');

%% Bar
figure1 = figure('Color',[1,1,1],'Position', [1 1 1200 300]);%[18 246 1332 420]
axes1 = axes('Parent',figure1);

% CPU
pCPU = y_iCPU + y_kCPU + y_aCPU;    % 100%
y_iCPUp = 100 * (y_iCPU ./pCPU);
y_kCPUp = 100 * (y_kCPU ./pCPU);
y_aCPUp = 100 * (y_aCPU ./pCPU);
y_CPU = [y_iCPUp, y_kCPUp, y_aCPUp];

% GPU
pGPU = y_iGPU + y_kGPU + y_aGPU;    % 100%
y_iGPUp = 100 * (y_iGPU ./pGPU);
y_kGPUp = 100 * (y_kGPU ./pGPU);
y_aGPUp = 100 * (y_aGPU ./pGPU);
y_GPU = [y_iGPUp, y_kGPUp, y_aGPUp];

% Fused
y = [y_CPU(1,:); y_GPU(1,:); ...
     y_CPU(2,:); y_GPU(2,:);...
     y_CPU(3,:); y_GPU(3,:);...
     y_CPU(4,:); y_GPU(4,:);...
     y_CPU(5,:); y_GPU(5,:);...
     y_CPU(6,:); y_GPU(6,:);...
     y_CPU(7,:); y_GPU(7,:);...
     y_CPU(8,:); y_GPU(8,:);...
     y_CPU(9,:); y_GPU(9,:);...
     y_CPU(10,:); y_GPU(10,:)];

% Plot bar
% b_CPU =bar(axes1, nel, [y_iCPUp'; y_kCPUp'; y_aCPUp'],'stacked');
nel2 = [nel-2.3; nel+2.3];
b = bar(nel2(:), y, 1, 'stacked' );

% % XTicks
% xtl = {{'CPU-GPU'; '10\times10\times10'},...
%        {'CPU-GPU'; '20\times20\times20'},...
%        {'CPU-GPU'; '30\times30\times30'},...
%        {'CPU-GPU'; '40\times40\times40'},...
%        {'CPU-GPU'; '50\times50\times50'},...
%        {'CPU-GPU'; '60\times60\times60'},...
%        {'CPU-GPU'; '70\times70\times70'},...
%        {'CPU-GPU'; '80\times80\times80'},...
%        {'CPU-GPU'; '90\times90\times90'},...
%        {'CPU-GPU'; '100\times100\times100'}};
%        
% h = my_xticklabels(gca,[10 20 30 40 50 60 70 80 90 100],xtl);
% 
set(axes1,'XTickLabel',...
    {'10\times10\times10','20\times20\times20','30\times30\times30','40\times40\times40','50\times50\times50','60\times60\times60','70\times70\times70','80\times80\times80','90\times90\times90','100\times100\times100'});

% Labels
xlabel('Number of elements (Left column: CPU. Right column: GPU)','FontSize',12,'Interpreter','latex');
ylabel('Percentage of total runtime','FontSize',12,'Interpreter','latex');

% Labels for indices
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints/2;
labels1 = [num2str(y(:,1),3), repmat('%',size(y,1),1)];
text(xtips1,ytips1,labels1,'FontSize',8,'FontWeight','bold',...
    'Interpreter','tex','Color','k',...
    'HorizontalAlignment','center', 'VerticalAlignment','middle');

% Labels for NI
xtips2 = b(2).XEndPoints;
ytips2 = (b(2).YEndPoints + b(1).YEndPoints)/2;
labels2 = [num2str(y(:,2),3), repmat('%',size(y,1),1)];
text(xtips2,ytips2,labels2,'FontSize',8,'FontWeight','bold',...
    'Interpreter','tex','Color','k',...
    'HorizontalAlignment','center', 'VerticalAlignment','middle');

% Labels for ASSEMBLY
xtips3 = b(2).XEndPoints;
ytips3 = (b(3).YEndPoints + b(2).YEndPoints)/2;
labels3 = [num2str(y(:,3),3), repmat('%',size(y,1),1)];
text(xtips3,ytips3,labels3,'FontSize',8,'FontWeight','bold',...
    'Interpreter','tex','Color','k',...
    'HorizontalAlignment','center', 'VerticalAlignment','middle');

% Create legend
legend('Index computation', 'Numerical integration', 'Assembly');
legend1 = legend(axes1,'show');
% title(legend1,'Legend')
set(legend1,'Orientation','horizontal','Location','northoutside','Interpreter','latex');


% Set the remaining axes properties
box(axes1,'on');
ylim(axes1,[0 100]);
xlim(axes1,[3 103]);





