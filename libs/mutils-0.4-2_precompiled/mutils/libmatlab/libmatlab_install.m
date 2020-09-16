function [installed, LIBMATLAB_OBJ] = libmatlab_install(basepath, config)
%LIBMATLAB_INSTALL compiles compiles the matlab interface libraries
%
%  [installed, LIBMATLAB_OBJ] = LIBMATLAB_INSTALL([basepath])
%
%Arguments:
%  basepath[=current directory]  : base installation path of libmatlab
%
%Output:
%  installed                     : 0 if failed, 1 if successful
%  LIBMATLAB_OBJ                 : compiled object files

% Copyright 2012, Marcin Krotkiewski, University of Oslo

if nargin==0
    basepath = pwd;
end
curpath = pwd;
chdir(basepath);

if exist('update_path')==2
    update_path(basepath);
else
    addpath(basepath);
    addpath([basepath filesep '..']);
end

if nargin<2
    config = mutils_config([basepath filesep '..']);
end

MEX_OBJ_SRC = [dir([basepath filesep '*.c']) ];
MEX_OBJ_SRC = cellfun(@strcat, repmat({[basepath filesep]}, 1, length(MEX_OBJ_SRC)), {MEX_OBJ_SRC.name},...
    'UniformOutput', false);
LIBMATLAB_OBJ = regexprep(MEX_OBJ_SRC, '\.c(pp)*$', config.obj_extension);


%% check, maybe we already do have what's needed in the path
installed = 1;
for i=1:numel(LIBMATLAB_OBJ)
    if ~exist(LIBMATLAB_OBJ{i}, 'file')
        installed = 0;
        break;
    end
end
if installed
    chdir(curpath);
    return;
end


%% Compile the mex files
try
    % compile objects
    config.mexflags{end+1} = '-c';
    for i=1:length(MEX_OBJ_SRC)
        fname =  MEX_OBJ_SRC{i};
        disp(['compiling ' regexprep(MEX_OBJ_SRC{i}, '\\', '\\\\')]);
        if ~isempty(ver('matlab'))
            % matlab
            mex(config.mexflags{:}, config.cflags, MEX_OBJ_SRC{i});
        else
            % octave
            setenv('CFLAGS', config.cflags);
            mex(config.mexflags{:}, MEX_OBJ_SRC{i});
            fflush(stdout);
        end
    end
    
    installed = 1;
catch
    warning([mfilename ': compilation of ' MEX_OBJ_SRC{i} ' failed.']);
    warning(lasterr);
end

chdir(curpath);

end
