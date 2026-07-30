function Plot_Index_runtime
%Plot_Index_runtime  Plot runtime results for INDEX performance tests.
%   Plot_Index_runtime loads precomputed performance data from
%   "IndexPerfTestOut_LNX.mat" and generates a log-log plot comparing
%   CPU and GPU runtimes for scalar and vector implementations with
%   32- and 64-bit integer element connectivity array.
%
%   The function creates a figure showing runtime (seconds) versus
%   number of finite elements (nel^3) for:
%       - CPU scalar 32-bit (CS32)
%       - GPU scalar 32-bit (GS32)
%       - CPU scalar 64-bit (CS64)
%       - GPU scalar 64-bit (GS64)
%       - CPU vector 32-bit (CV32)
%       - GPU vector 32-bit (GV32)
%       - CPU vector 64-bit (CV64)
%       - GPU vector 64-bit (GV64)
%
% The function expects the MAT-file to contain variables used below such
%   as fullTable, nel_all, dTEall, dTNall, prob_all, and proc_all. It builds
%   parameter arrays in the same nested-loop order as used to produce
%   fullTable and issues a warning if the number of rows in fullTable does
%   not match the expected total runs.
% 
%   Example:
%       Plot_Index_runtime
%

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Institución Universitaria Pascual Bravo, Medellin-Colombia
%   Created: July 30, 2026. Version: 1.0

%% Data
load("IndexPerfTestOut_LNX2026.mat"); %#ok

% Build parameter arrays in the same nested-loop order used to produce fullTable
nN = numel(nel_all);
nTE = numel(dTEall);
nTN = numel(dTNall);
nP  = numel(prob_all);
nPr = numel(proc_all);

totalRuns = nN * nTE * nTN * nP * nPr;
nRows = height(fullTable);
if nRows ~= totalRuns
    warning('fullTable rows (%d) do not match expected runs (%d).', nRows, totalRuns);
end

stride = totalRuns / nN;
nels = nel_all.^3;

%% Figure
fig = figure('color','none',Name='Index runtime');
ax  = axes('Parent',fig,'Color','none','Box','on');

pt = plot(ax, ... 
    nels, fullTable.Mean(1:stride:end),'--ob',...   % CPU_Scalar_uint32
    nels, fullTable.Mean(2:stride:end),'-ob',...   % GPU_Scalar_uint32
    nels, fullTable.Mean(3:stride:end),'--+b',...  % CPU_Vector_uint32
    nels, fullTable.Mean(4:stride:end),'-+b',...  % GPU_Vector_uint32
    nels, fullTable.Mean(5:stride:end),'--or',...   % CPU_Scalar_uint64
    nels, fullTable.Mean(6:stride:end),'-or',...   % GPU_Scalar_uint64
    nels, fullTable.Mean(7:stride:end),'--+r',...  % CPU_Vector_uint64
    nels, fullTable.Mean(8:stride:end),'-+r');    % GPU_Vector_uint64

% Convection:
%   --: for CPU (dashed line)
%   -:  for GPU (solid line)
%   o:  for Scalar (marker o)
%   +:  for Vector (marker +)
%   b:  for uint32 (blue line)
%   r:  for uint64 (red line)

% LineWidth
% nl = size(pt,1);
% for i=1:nl
%     pt(i).LineWidth = 2;
% end

inter = 'latex';

% Labels
xlabel(ax,'Number of finite elements','Interpreter',inter);
ylabel(ax, 'Runtime (s)','Interpreter',inter);

% Create legend
lg = legend(ax,...
    {'CS32','GS32',...
    'CV32','GV32',...
    'CS64','GS64',...
    'CV64','GV64'});
% lg.FontSize = 14;
lg.NumColumns = 2;
lg.Location = 'northwest';
lg.Interpreter = inter;

% Set the remaining axes properties
% axis(ax,'tight','square');
set(ax,'XGrid','on','XMinorTick','on','XScale','log',...
    'YGrid','on','YMinorTick','on','YScale','log','FontSize',20,'TickLabelInterpreter',inter);

%% save as PDF
% exportgraphics(gcf,'IndexRuntime2026.pdf','BackgroundColor','none','ContentType','vector')