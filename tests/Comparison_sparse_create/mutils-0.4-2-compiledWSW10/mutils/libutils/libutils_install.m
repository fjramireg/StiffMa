function [installed LIBUTILS_OBJ] = libutils_install(basepath, config)
%LIBUTILS_INSTALL compiles compiles the utility libraries
%
%  [installed, LIBUTILS_OBJ] = LIBUTILS_INSTALL([basepath])
%
%Arguments:
%  basepath[=current directory]  : base installation path of libutils
%
%Output:
%  installed                     : 0 if failed, 1 if successful
%  LIBUTILS_OBJ                  : compiled object files

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

%% check, maybe we already do have what's needed in the path
% find all C files in libutils
LIBUTILS = dir([basepath filesep '*.c']);
LIBUTILS = cellfun(@strcat, repmat({[basepath filesep]}, 1, length(LIBUTILS)), {LIBUTILS.name},...
    'UniformOutput', false);
LIBUTILS_OBJ = regexprep(LIBUTILS, '\.c$', config.obj_extension);

% check if already compiled
installed = 1;
for i=1:numel(LIBUTILS_OBJ)
    if ~exist(LIBUTILS_OBJ{i}, 'file')
        installed = 0;
        break;
    end
end
if installed
    chdir(curpath);
    return;
end

%% Compile object files of LIBUTILS
try
    config.mexflags{end+1} = '-c';
    for i=1:length(LIBUTILS)
        disp(['compiling ' regexprep(LIBUTILS{i}, '\\', '\\\\')]);
        if ~isempty(ver('matlab'))
            % matlab
            mex(config.mexflags{:}, config.cflags, LIBUTILS{i});
        else
            % octave
            setenv('CFLAGS', config.cflags);
            mex(config.mexflags{:}, LIBUTILS{i});
            fflush(stdout);
        end
    end
    installed = 1;
catch
    warning([mfilename ': compilation of ' LIBUTILS{i} ' failed.']);
    warning(lasterr);
end

chdir(curpath);

end
