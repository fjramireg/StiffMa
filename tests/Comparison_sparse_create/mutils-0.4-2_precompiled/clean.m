function clean
%CLEAN removes compiled MEX functions from mutils tools and external packages.
%This is needed in case you want to recompile mutils. This function must be
%called from the top mutils directory. In MATLAB:
%
%  >> cd path/to/mutils
%  >> clean


%% Check if we are running inside milamin, or independently
global milamin_data;
if ~isfield(milamin_data, 'path')
    basepath = pwd;
else
    basepath = [milamin_data.path filesep 'ext'];
end

%% remove old mex files
SUBDIRS = {'SuiteSparse', 'triangle', ...
    ['mutils' filesep 'quadtree'], ...
    ['mutils' filesep 'interp'], ['mutils' filesep 'reorder'], ...
    ['mutils' filesep 'sparse']};

for i=1:numel(SUBDIRS)
    delete([SUBDIRS{i} filesep '*.' mexext]);    
end
