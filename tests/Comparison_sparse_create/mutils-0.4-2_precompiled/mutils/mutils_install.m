function installed = mutils_install(basepath)
%MUTILS_INSTALL compiles the mutils components and adds necessary paths to MATLAB environment
%
%  installed = MUTILS_INSTALL([basepath])
%
%Arguments:
%  basepath[=current directory]  : base installation path of mutils
%
%Output:
%  installed                     : 0 if failed, 1 if successful

% Copyright 2012, Marcin Krotkiewski, University of Oslo
if nargin==0
    basepath = pwd;
end
curpath=pwd;
cd(basepath);

if exist('update_path')==2
  update_path(basepath);
else
  addpath(basepath);
end

% mutils_clean(basepath);
installed = 1;

SUBDIRS = {'quadtree', 'interp', 'reorder', 'sparse'};
for i=1:numel(SUBDIRS)
    d = [basepath filesep SUBDIRS{i}];
    cd(d);
    instr=sprintf('%s_install', lower(SUBDIRS{i}));
    try
        func = str2func(instr);
        res = func(d);
        if res
            disp([SUBDIRS{i} ' compiled successfully']);
        else
            disp(['ERROR while compiling ' SUBDIRS{i}]);
            installed = 0;
        end
    catch 
        disp(['ERROR while compiling ' SUBDIRS{i}]);
        disp(lasterr);
        installed = 0;
    end
    disp(' ');
    cd(basepath)
end

cd(curpath);

end
