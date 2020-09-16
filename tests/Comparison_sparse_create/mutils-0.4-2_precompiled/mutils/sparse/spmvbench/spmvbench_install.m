function [installed OBJ] = spmvbench_install(basepath)

% Copyright 2012, Marcin Krotkiewski, University of Oslo

if nargin==0
    basepath = pwd;
end
curpath = pwd;
chdir(basepath);

config = mutils_config([basepath filesep '../..']);

%% check, maybe we already do have what's needed in the path
% find all C files in libutils
SRC = dir([basepath filesep '*.c']);
SRC = cellfun(@strcat, repmat({[basepath filesep]}, 1, length(SRC)), {SRC.name},...
    'UniformOutput', false);
OBJ = regexprep(SRC, '\.c$', config.obj_extension);

s=warning('query', 'MATLAB:DELETE:FileNotFound');
warning('off', 'MATLAB:DELETE:FileNotFound');
delete(OBJ{:});
warning(s.state, 'MATLAB:DELETE:FileNotFound');

% check if already compiled
installed = 1;
for i=1:numel(OBJ)
    if ~exist(OBJ{i}, 'file')
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
    for i=1:length(SRC)
        display(['compiling ' regexprep(SRC{i}, '\\', '\\\\')]);
        mex(config.mexflags{:}, config.cflags, config.coptimflags, SRC{i});
    end
    installed = 1;
catch
    warning([mfilename ': compilation of spmvbench failed.']);
end

chdir(curpath);

end
