function Plot_Index_speedup
%Plot_Index_speedup Plot CPU/GPU index operation speedup vs. problem size.
%   Plot_Index_speedup loads precomputed benchmark results from
%   "IndexPerfTestOut_LNX2026.mat" and produces a log-log plot showing the
%   speedup ratios (CPU time / GPU time) for four cases:
%       - Scalar uint32
%       - Vector uint32
%       - Scalar uint64
%       - Vector uint64
%
%   The function expects the MAT-file to contain variables used below such
%   as fullTable, nel_all, dTEall, dTNall, prob_all, and proc_all. It builds
%   parameter arrays in the same nested-loop order as used to produce
%   fullTable and issues a warning if the number of rows in fullTable does
%   not match the expected total runs.
%
%   The produced figure displays:
%     - x-axis: number of finite elements (nel_all.^3)
%     - y-axis: speedup ratio (CPU / GPU)
%     - legend entries: 'S32','V32','S64','V64'
%
%   No inputs or outputs. Designed to be run as a script-like function that
%   creates and displays the figure.
%
%   Example:
%       Plot_Index_speedup
%

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Institución Universitaria Pascual Bravo, Medellin-Colombia
%   Created: July 30, 2026. Version: 1.0

%% Load data
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
fig = figure('color','none',Name='GPU speedup (index)');
ax  = axes('Parent',fig,'Color','none','Box','on');

plot(ax, ...
    nels, fullTable.Mean(1:stride:end)./fullTable.Mean(2:stride:end),'-ob',...  % CPU_Scalar_uint32 / GPU_Scalar_uint32
    nels, fullTable.Mean(3:stride:end)./fullTable.Mean(4:stride:end),'-+b',...  % CPU_Vector_uint32 / GPU_Vector_uint32
    nels, fullTable.Mean(5:stride:end)./fullTable.Mean(6:stride:end),'-or',...  % CPU_Scalar_uint64 / GPU_Scalar_uint64
    nels, fullTable.Mean(7:stride:end)./fullTable.Mean(8:stride:end),'-+r');    % CPU_Vector_uint64 / GPU_Vector_uint64

inter = 'latex';

% Labels
xlabel(ax,'Number of finite elements','Interpreter',inter);
ylabel(ax, 'Speedup ratio','Interpreter',inter);

% Create legend
lg = legend(ax, {'S32','V32','S64','V64'});
% lg.FontSize = 14;
% lg.NumColumns = 2;
lg.Location = 'best';
lg.Interpreter = inter;

% Set the remaining axes properties
% axis(ax,'tight','square');
set(ax,'XGrid','on','XMinorTick','on','XScale','log',...
    'YGrid','on','YMinorTick','on','YScale','log','FontSize',20,'TickLabelInterpreter',inter);

%% save as PDF
% exportgraphics(gcf,'IndexSpeedup2026.pdf','BackgroundColor','none','ContentType','vector')