function install
%INSTALL mutils tools and external packages. See INSTALL and RELEASE_NOTES for details
%
%Currently installed components:
% - a number of mex functions from SuiteSparse
% - triangle mex function
% - core mutils components: 
%     o quadtree
%     o einterp
%     o sparse_create
%     o sparse_convert
%     o spmv
%     o metis_part
%     o mrcm

% Copyright 2012, Marcin Krotkiewski, University of Oslo

%% Check if we are running inside milamin, or independently
global milamin_data;
if ~isfield(milamin_data, 'path')
    basepath = pwd;
else
    basepath = [milamin_data.path filesep 'ext'];
end
curpath  = pwd;

if exist('update_path')==2
    update_path(basepath);
else
    addpath(basepath);
end

%% Check if we know how to compile
try
    cd(basepath);
    fid=fopen('test.c', 'w+');
    if fid==-1
        warning([mfilename ': MATLAB MEX compiler can not be tested. Trying anyhow.']);
    else
        fprintf(fid, 'int mexFunction(){return 0;}\n');
        fclose(fid);
        mex('test.c');
        loc = dir(['test.' mexext]);
        if isempty(loc)
            warning([mfilename ': MATLAB MEX compiler is not configured correctly.'...
                ' Cannot compile a test program. MILAMIN will not use external packages.']);
            return;
        end
        delete('test.c');
        delete(['test.' mexext]);
    end
catch
    wrnstate = warning('query', 'MATLAB:DELETE:FileNotFound');
    warning('off', 'MATLAB:DELETE:FileNotFound');
    delete('test.c');
    delete(['test.' mexext]);
    warning(wrnstate);
    warning([mfilename ': MATLAB MEX compiler is not configured correctly.'...
        ' Cannot compile test program. MILAMIN will not use external packages.']);
    cd(curpath);
    return;
end

%% Compile/Install external packages
clear global mutils_paths;
update_path([basepath]);

installed = 1;
SUBDIRS = {'triangle', 'SuiteSparse', 'mutils'};
for i=1:numel(SUBDIRS)
    d = [basepath filesep SUBDIRS{i}];
    cd(d);
    instr=sprintf('%s_install', lower(SUBDIRS{i}));
    try
        func = str2func(instr);
        res = func(d);
        if res
            disp([SUBDIRS{i} ' available']);
        else
            disp([SUBDIRS{i} ' NOT available.']);
            installed = 0;
        end
    catch
        disp([SUBDIRS{i} ' NOT available.']);
        disp(lasterr);
        installed = 0;
    end
    disp(' ');
    cd(basepath)
end

% print addpath info
disp(' ');
disp('--------------------------------------------------------------------');
disp('Paths required by mutils have been added to your MATLAB environment.');
disp('You need to either save the path from menu File->Set Path,');
disp('or add the following lines to your code whenever you want to use');
disp('mutils:');
disp(' ');

global mutils_paths;
for i=1:length(mutils_paths)
    disp(['    addpath(''' mutils_paths{i} ''')']);
end

if ~installed
    disp(' ');
    disp('WARNING: There have been errors during compilation of some MUTILS components.');
    disp('WARNING: Not all modules are available!');
end

end
