function installed = sparse_install(basepath)
%SPARSE_INSTALL compiles compiles sparse MEX utilities
%
%  installed = SPARSE_INSTALL([basepath])
%
%Arguments:
%  basepath[=current directory]  : base installation path of sparse
%
%Output:
%  installed                     : 0 if failed, 1 if successful
%
%See also: SPARSE_CONVERT, SPMV, SPARSE_INFO

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
MEX_FUNCTIONS = {'sparse_convert' 'spmv' 'sparse_info'};
MEX_SRC = {'sparse_convert_mex.c' 'spmv_mex.c' 'sparse_info_mex.c'};
LIBSPARSE = {'sparse_opts.c', 'mexio.c', 'sparse_utils.c', 'comm.c', 'sp_matv.c'};
LIBUTILS_OBJ = regexprep(LIBSPARSE, '\.c$', config.obj_extension);

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
[status, MEX_OBJ] = libutils_install(pwd);
cd(basepath);


%% MATLAB utility functions
cd([basepath filesep '..' filesep 'libmatlab']);
[status, OBJ] = libmatlab_install(pwd);
cd(basepath);
MEX_OBJ = [MEX_OBJ OBJ];


%% Compile the object files
try
    mexflags = config.mexflags;
    mexflags{end+1} = '-c';
    for i=1:length(LIBSPARSE)
        disp(['compiling ' regexprep(LIBSPARSE{i}, '\\', '\\\\')]);
        if ~isempty(ver('matlab'))
            % matlab
            mex(mexflags{:}, config.cflags, LIBSPARSE{i});
        else
            % octave
            setenv('CFLAGS', config.cflags);
            mex(mexflags{:}, LIBSPARSE{i});
            fflush(stdout);
        end
    end
    MEX_OBJ = [MEX_OBJ LIBUTILS_OBJ];
catch
    warning([mfilename ': compilation of sparse tools failed.']);
    warning(lasterr);
    chdir(curpath);
    return;
end


%% Compile the mex files
try
    for i=1:length(MEX_FUNCTIONS)
        disp(['compiling ' MEX_SRC{i}]);
        ofname = MEX_FUNCTIONS{i};
        if ~isempty(ver('matlab'))
            % matlab
            mex(config.mexflags{:}, config.cflags, MEX_SRC{i}, MEX_OBJ{:}, config.ldflags, config.mex_output, ofname);
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

% install sparse_create
sparse_create_install(basepath);

end
