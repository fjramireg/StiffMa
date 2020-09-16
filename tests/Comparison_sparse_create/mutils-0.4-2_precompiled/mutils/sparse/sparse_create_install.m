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
%See also: SPARSE_CREATE, SPARSE_CONVERT, SPMV

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

% sparse_create is a special case. tcmalloc is used as memory allocator.
% For that, libutils need to be recompiled with tcmalloc
config = mutils_config([basepath filesep '..']);
config.cflags = [config.cflags config.tcmalloc_flags];

% clean incompatible objects
SUBDIRS = {'.', ['..' filesep 'libutils'], ['..' filesep 'libmatlab']};
for i=1:numel(SUBDIRS)
    delete([basepath filesep SUBDIRS{i} filesep '*' config.obj_extension]);
end


% list of mex functions
MEX_FUNCTIONS = {'sparse_create'};
MEX_SRC = {'sparse_create_mex.c'};
LIBSPARSE = {'sparse_opts.c'};
LIBUTILS_OBJ = regexprep(LIBSPARSE, '\.c$', config.obj_extension);

%% check, maybe we already do have what's needed in the path
for i=1:numel(MEX_FUNCTIONS)
    if exist([MEX_FUNCTIONS{i} '.' mexext]) == 3
        warning(['Old version of ' MEX_FUNCTIONS{i} '.' mexext ' already installed on this system will be ignored']);
    end
end


%% Compile object files of LIBUTILS
cd([basepath filesep '..' filesep 'libutils']);
[status, MEX_OBJ] = libutils_install(pwd, config);
cd(basepath);


%% MATLAB utility functions
cd([basepath filesep '..' filesep 'libmatlab']);
[status, OBJ] = libmatlab_install(pwd, config);
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

    % clean incompatible objects
    SUBDIRS = {'.', ['..' filesep 'libutils'], ['..' filesep 'libmatlab']};
    for i=1:numel(SUBDIRS)
        delete([basepath filesep SUBDIRS{i} filesep '*' config.obj_extension]);
    end
    
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
            if ~isempty(config.tcmalloc_libs)
                mex(config.mexflags{:}, config.cflags, MEX_SRC{i}, MEX_OBJ{:}, config.ldflags, config.mex_output, ofname, config.tcmalloc_libs);
            else
                mex(config.mexflags{:}, config.cflags, MEX_SRC{i}, MEX_OBJ{:}, config.ldflags, config.mex_output, ofname);
            end
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

% clean incompatible objects
SUBDIRS = {'.', ['..' filesep 'libutils'], ['..' filesep 'libmatlab']};
for i=1:numel(SUBDIRS)
    delete([basepath filesep SUBDIRS{i} filesep '*' config.obj_extension]);
end

chdir(curpath);

end
