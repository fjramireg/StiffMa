function installed = quadtree_install(basepath)
%QUADTREE_INSTALL compiles compiles the quadtree MEX function
%
%  installed = QUADTREE_INSTALL([basepath])
%
%Arguments:
%  basepath[=current directory]  : base installation path of quadtree
%
%Output:
%  installed                     : 0 if failed, 1 if successful
%
%See also: QUADTREE, TSEARCH2

% Copyright 2012, Marcin Krotkiewski, University of Oslo

installed = 0;
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

config = mutils_config([basepath filesep '..']);

% list of mex functions
MEX_FUNCTIONS = {'quadtree' 'octree'};
MEX_SRC = {'octree_mex.c' 'octree_mex.c'};
SRC = {'octree_opts.c'};
MEX_CFLAGS = {' -DNDIM=2' ' -DNDIM=3'};
MEX_OBJ = regexprep(SRC, '\.c$', config.obj_extension);

%% check, maybe we already do have what's needed in the path
installed = 1;
for i=1:numel(MEX_FUNCTIONS)
    if isempty(dir([MEX_FUNCTIONS{i} '.' mexext]))
        installed = 0;
        if exist([MEX_FUNCTIONS{i} '.' mexext]) == 3
            warning(['Old version of ' MEX_FUNCTIONS{i} '.' mexext ' already installed on this system will be ignored']);
        end
    end
end

if installed
    return;
end


%% Compile object files of LIBUTILS
cd([basepath filesep '..' filesep 'libutils']);
[status, OBJ] = libutils_install(pwd);
cd(basepath);
MEX_OBJ = [MEX_OBJ OBJ];


%% MATLAB utility functions
cd([basepath filesep '..' filesep 'libmatlab']);
[status, OBJ] = libmatlab_install(pwd);
cd(basepath);
MEX_OBJ = [MEX_OBJ OBJ];


%% Compile the mex files
try
    mexflags = config.mexflags;
    mexflags{end+1} = '-c';
    for i=1:length(SRC)
        fname = SRC{i};
        disp(['compiling ' regexprep(SRC{i}, '\\', '\\\\')]);
        if ~isempty(ver('matlab'))
            % matlab
            mex(mexflags{:}, config.coptimflags, config.cflags, SRC{i});
        else
            % octave
            setenv('CFLAGS', config.cflags);
            mex(mexflags{:}, SRC{i});
            fflush(stdout);
        end
    end
    
    for i=1:length(MEX_FUNCTIONS)
        disp(['compiling ' regexprep(MEX_SRC{i}, '\\', '\\\\')]);
        ofname = MEX_FUNCTIONS{i};
        if ~isempty(ver('matlab'))
            % matlab
            mex(config.mexflags{:}, [config.cflags MEX_CFLAGS{i}], MEX_SRC{i}, MEX_OBJ{:}, config.ldflags, config.mex_output, ofname);
        else
            % octave
            setenv('CFLAGS', config.cflags);
            setenv('LDFLAGS', config.ldflags);
            mex(config.mexflags{:}, MEX_SRC{i}, MEX_OBJ{:}, config.mex_output, [ofname '.' mexext]);
            fflush(stdout);
        end
    end
    installed = 1;
catch
    warning([mfilename ': compilation of ' MEX_SRC{i} ' failed.']);
    warning(lasterr);
end

chdir(curpath);

end
