% function Plot_Index_runtimeMaxMeshSize

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Institución Universitaria Pascual Bravo, Medellin-Colombia
%   Created: July 30, 2026. Version: 1.0

%% Data
load("IndexPerfTestOut_LNXMax2026.mat"); 

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
% nels = nel_all.^3;
nels = nel_all;

%% Figure
fig = figure('color','none',Name='Index runtime (Max. mesh size)');
ax  = axes('Parent',fig,'Color','none','Box','on');

h = plot(ax, ...
    nels, fullTable.Mean(1:stride:end),'-ob',...  % GPU_Scalar_uint32
    nels, fullTable.Mean(2:stride:end),'-+b',...  % GPU_Vector_uint32
    nels, fullTable.Mean(3:stride:end),'-or',...  % GPU_Scalar_uint64
    nels, fullTable.Mean(4:stride:end),'-+r');    % GPU_Vector_uint64

% Convection:
%   --: for CPU (dashed line)
%   -:  for GPU (solid line)
%   o:  for Scalar (marker o)
%   +:  for Vector (marker +)
%   b:  for uint32 (blue line)
%   r:  for uint64 (red line)

inter = 'latex';

% Add datatip at the last finite (non-NaN) point of each curve
idxs = zeros(numel(h),1);
for ii = 1:numel(h)
    xdata = h(ii).XData;
    ydata = h(ii).YData;
    idx = find(isfinite(ydata), 1, 'last');     % last valid numeric entry
    idxs(ii) = idx;
    if isempty(idx)
        continue
    end
    % Create a datatip at (xdata(idx), ydata(idx))
    dt = datatip(h(ii), xdata(idx), ydata(idx)); % R2020b+
    % dt = datatip(h(ii), xdata(idx), ydata(idx),nels(idx)); % R2020b+
    % dt = datatip(h(ii), idx); % R2020b+

    % Optional styling
    % dt.FontSize = 12;
    dt.Interpreter = inter;           % use your existing 'inter' (e.g. 'latex')
    % dt.LineStyle = 'none';            % remove connecting line in the datatip
    % dt.BackgroundColor = 'none';
end

% Labels
xlabel(ax,'Number of finite elements ($nel^3$)','Interpreter',inter);
ylabel(ax, 'Runtime (s)','Interpreter',inter);

% Create legend
lg = legend(ax,{'GS32','GV32','GS64','GV64'});
% lg.FontSize = 14;
lg.NumColumns = 2;
lg.Location = 'best';
lg.Interpreter = inter;

% Set the remaining axes properties
% axis(ax,'tight','square');
set(ax,'XGrid','on','XMinorTick','on','XScale','lin',...
    'YGrid','on','YMinorTick','on','YScale','lin','FontSize',20,'TickLabelInterpreter',inter);

%% save as PDF
% exportgraphics(gcf,'IndexRuntime2026.pdf','BackgroundColor','none','ContentType','vector')