function mutils_clean(basepath)
%MUTILS_CLEAN deletes all compiled object files and mex files of mutils
%
%  MUTILS_CLEAN([basepath])
%
%Arguments:
%  basepath[=current directory]  : base installation path of mutils

% Copyright 2012, Marcin Krotkiewski, University of Oslo

if nargin==0
    basepath = pwd;
end

config = mutils_config(basepath);

SUBDIRS = {'', 'libutils', 'libmatlab', 'reorder', 'sparse', 'quadtree', 'interp'};
for i=1:numel(SUBDIRS)
    delete([basepath filesep SUBDIRS{i} filesep '*' config.obj_extension]);
    delete([basepath filesep SUBDIRS{i} filesep '*.' mexext]);
end


end
