function installed = suitesparse_install(basepath)
%SUITESPARSE_INSTALL downloads and installs parts of SuiteSparse by Tim Davis
%
%  installed = SUITESPARSE_INSTALL([basepath])
%
%Arguments:
%  basepath[=current directory]  : base installation path of SuiteSparse
%
%Output:
%  installed                     : 0 if failed, 1 if successful
%

% Copyright 2012, Marcin Krotkiewski, University of Oslo

if nargin==0
    basepath = pwd;
end

if exist('update_path')==2
  update_path(basepath);
else
  addpath(basepath);
end

% check, maybe we already do have what's needed in the path
if exist(['metis.' mexext]) == 3 &...
   exist(['cs_lsolve.' mexext]) == 3 &...
   exist(['cs_ltsolve.' mexext]) == 3 &...
   exist(['cs_permute.' mexext]) == 3 &...
   exist(['cs_symperm.' mexext]) == 3 &...
   exist(['cs_transpose.' mexext]) == 3 &...
   exist(['sparse2.' mexext]) == 3 &...
   exist(['lchol.' mexext]) == 3
    disp('All required SuiteSparse components are already installed on this system.');
    disp('Using existing MEX files.');
    installed = 1;
    return;
end

suitesparse_url = 'http://www.cise.ufl.edu/research/sparse/SuiteSparse/SuiteSparse-4.0.2.tar.gz';
metis_url = 'http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/OLD/metis-4.0.3.tar.gz';
srcpath   = [basepath filesep 'sources' ];
cholpath  = [srcpath filesep 'CHOLMOD' filesep 'MATLAB' filesep];
cspath    = [srcpath filesep 'CXSparse' filesep 'MATLAB' filesep 'CSparse' filesep];
curpath   = pwd;
installed = 0;
cd(basepath);


%% Download and unzip
if ~exist(srcpath)
    
    disp([mfilename ': attempting to download and install SuiteSparse']);
    ofname = [basepath filesep 'SuiteSparse.tgz'];
    if exist(ofname)
        status = 1;
        nodelete = 1;
    else
        try
            [f,status] = urlwrite(suitesparse_url, ofname);
            nodelete = 0;
        catch err
            disp(['Error downloading SuiteSparse: ' err.message]);
            status = 0;
        end
    end
    if status==0
        warning([mfilename ': could not download SuiteSparse. MILAMIN will not be able to use SuiteSparse and METIS.']);
        cd(curpath);
        return;
    end
    untar(ofname, basepath);
    movefile([basepath filesep 'SuiteSparse'], srcpath);
    if ~nodelete
        delete(ofname);
    end
    
    disp([mfilename ': attempting to download and install Metis']);
    ofname = [basepath filesep 'metis.tgz'];
    if exist(ofname)
        status = 1;
        nodelete = 1;
    else
        try
            [f,status] = urlwrite(metis_url, ofname);
            nodelete = 0;
        catch err
            disp(['Error downloading METIS: ' err.message]);
            status = 0;
        end
    end
    if status==0
        warning([mfilename ': could not download Metis. MILAMIN will not be able to use SuiteSparse and METIS.']);
        return;
    end
    untar(ofname, srcpath);
    movefile([srcpath filesep 'metis-4.0.3'], [srcpath filesep 'metis-4.0']);
    if ~nodelete
        delete(ofname);
    end
    delete_sources = 1;
    
    % apply 64-bit patch for METIS 4
    copyfile('metis.h', [srcpath filesep 'metis-4.0' filesep 'Lib']);
    copyfile('struct.h', [srcpath filesep 'metis-4.0' filesep 'Lib']);
    copyfile('proto.h', [srcpath filesep 'metis-4.0' filesep 'Lib']);
    copyfile('minitpart.c', [srcpath filesep 'metis-4.0' filesep 'Lib']);
    copyfile('mmd.c', [srcpath filesep 'metis-4.0' filesep 'Lib']);
    copyfile('cholmod_metis.c', [srcpath filesep 'CHOLMOD' filesep 'Partition']);
    copyfile('cholmod_matlab.h', [srcpath filesep 'CHOLMOD' filesep 'MATLAB']);
    copyfile('cholmod_make.m', [srcpath filesep 'CHOLMOD' filesep 'MATLAB']);
    
else
    delete_sources = 0;
end

installed = 1;

%% Compile CHOLMOD and the metis module
if exist(['metis.' mexext]) ~= 3 |...
   exist(['sparse2.' mexext]) ~= 3 |...
   exist(['lchol.' mexext]) ~= 3
    try
        %fix compilation problems in SuiteSparse-4.0.0
        %copyfile('cholmod_make.m', cholpath);
        cd(cholpath);
        cholmod_make;
        movefile([cholpath filesep 'sparse2.' mexext], basepath);
        movefile([cholpath filesep 'lchol.' mexext], basepath);
        movefile([cholpath filesep 'metis.' mexext], basepath);
    catch
        warning([mfilename ': compilation of SuiteSparse failed. MILAMIN will not be able to use METIS.']);
        installed = 0;
    end
end

% compile cs_ltsolve etc.
if exist(['cs_lsolve.' mexext])  ~= 3 |...
   exist(['cs_ltsolve.' mexext]) ~= 3 |...
   exist(['cs_transpose.' mexext]) ~= 3 |...
   exist(['cs_permute.' mexext]) ~= 3 |...
   exist(['cs_symperm.' mexext]) ~= 3
    try
        cd(cspath);
        cs_make;
        movefile([cspath filesep 'cs_ltsolve.' mexext], basepath);
        movefile([cspath filesep 'cs_lsolve.' mexext], basepath);
        movefile([cspath filesep 'cs_permute.' mexext], basepath);
        movefile([cspath filesep 'cs_symperm.' mexext], basepath);
        movefile([cspath filesep 'cs_transpose.' mexext], basepath);
        installed = 1 & installed;
    catch
        warning([mfilename ': compilation of CXSparse failed. MILAMIN will not be able to use cs_ltsolve.']);
        installed = 0;
    end
end

cd(basepath);
if delete_sources
    rmdir(srcpath, 's');
end
cd(curpath);

end
