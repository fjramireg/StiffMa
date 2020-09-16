function installed = triangle_install(basepath)
%TRIANGLE_INSTALL downloads triangle mesh generator by Jonathan Shewchuk
%The sources are compiled with a MEX file wrapper.
%
%  installed = TRIANGLE_INSTALL([basepath])
%
%Arguments:
%  basepath[=current directory]  : base installation path of triangle
%
%Output:
%  installed                     : 0 if failed, 1 if successful

% Copyright 2012, Marcin Krotkiewski, University of Oslo

if nargin==0
    basepath = pwd;
end

if exist('update_path')==2
  update_path(basepath);
else
  addpath(basepath);
end

%% check, maybe we already do have what's needed in the path
if exist(['triangle.' mexext]) == 3
    display('triangle MEX file is already installed on this system.');
    display('Using existing MEX file.');
    installed = 1;
    return;
end

config = compiler_flags;

external_url = 'http://www.netlib.org/voronoi/triangle.zip';
archivename  = 'triangle.zip';
srcpath      = [basepath filesep 'sources' ];
curpath      = pwd;
installed    = 0;
cd(basepath);

%% download and unzip triangle sources
if ~exist(srcpath)
    
    display([mfilename ': attempting to download and install triangle mesh generator']);
    ofname = [basepath filesep 'triangle.zip'];
    if exist(ofname)
        status = 1;
        nodelete = 1;
    else
        try
            [f,status] = urlwrite(external_url, ofname);
            nodelete = 0;
        catch err
            display(['Error downloading triangle: ' err.message]);
            status = 0;
        end
    end
    if status==0
        cd(curpath);
        warning([mfilename ': could not download triangle.']);
        return;
    end
    if ~exist(srcpath)
        mkdir(srcpath);
    end
    unzip(ofname, srcpath);
    if ~nodelete
        delete(ofname);
    end    
    delete_sources = 1;
else
    delete_sources = 0;
end

%% Compile triangle mex file
try    
    mexflags = {'-DTRILIBRARY' '-DNO_TIMER' ['-I' srcpath]};
    if isfield(config,'mexflags')
        mexflags{end+1} = config.mexflags;
    end
    mex(mexflags{:}, 'triangle_mex.c', config.mex_output, ['triangle.' mexext], ['sources' filesep 'triangle.c']);
    installed = 1;
catch
    warning([mfilename ': compilation of triangle failed:' lasterr]);
end

cd(basepath);
if delete_sources
    rmdir(srcpath, 's');
end
cd(curpath);

end
