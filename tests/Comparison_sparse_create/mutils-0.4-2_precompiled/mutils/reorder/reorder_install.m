function installed = reorder_install(basepath)
%REORDER_INSTALL compiles a number of matrix reordering MEX files
%
%  installed = REORDER_INSTALL([basepath])
%
%Currently, the module provides an interface to METIS graph partioning
%library and a fast implementation of RCM bandwidth-reducing reordering.
%The installation script downloads and compiles version 4 of METIS.
%
%Arguments:
%  basepath[=current directory]  : base installation path of reorder
%
%Output:
%  installed                     : 0 if failed, 1 if successful
%
%See also: METIS_PART, MRCM

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

% disable all compiler warnings for METIS
config_metis = mutils_config([basepath filesep '..'],1);
config = mutils_config([basepath filesep '..']);

% list of mex functions
MEX_FUNCTIONS = {'metis_part', 'mrcm'};
MEX_SRC = {'metis_part_mex.c', 'mrcm_mex.c'};
SRC = {'reorder_metis.c', 'reorder_rcm.c'};
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


metis_url = 'http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/OLD/metis-4.0.3.tar.gz';
metispath = [basepath filesep 'sources'];

%% Download and unzip
if ~exist(metispath)
    
    disp([mfilename ': attempting to download and compile Metis']);
    ofname = [basepath filesep 'metis.tgz'];
    if exist(ofname)
        status = 1;
        nodelete = 1;
    else
        try
            [f,status] = urlwrite(metis_url, ofname);
            nodelete = 0;
        catch
            disp(['Error downloading METIS: ' lasterr]);
            status = 0;
        end
    end
    if status==0
        warning([mfilename ': could not download Metis. MILAMIN will not be able to use SuiteSparse and METIS.']);
        return;
    end
    untar(ofname, basepath);
    movefile([basepath filesep 'metis-4.0.3'], metispath);
    if ~nodelete
        delete(ofname);
    end
    
    % apply 64-bit patch for METIS 4
    copyfile('metis.h', [metispath filesep 'Lib']);
    copyfile('struct.h', [metispath filesep 'Lib']);
    copyfile('proto.h', [metispath filesep 'Lib']);
    copyfile('minitpart.c', [metispath filesep 'Lib']);
    copyfile('mmd.c', [metispath filesep 'Lib']);
end

% list source files
LIBMETIS = dir([metispath filesep 'Lib' filesep '*.c']);
LIBMETIS = cellfun(@strcat, repmat({[metispath filesep 'Lib' filesep]}, 1, length(LIBMETIS)), {LIBMETIS.name},...
    'UniformOutput', false);
LIBMETIS_OBJ = regexprep(LIBMETIS, '\.c$', config.obj_extension);

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

% check if METIS sources are already compiled
installed = 1;
for i=1:numel(LIBMETIS_OBJ)
    if ~exist(LIBMETIS_OBJ{i}, 'file')
        installed = 0;
        break;
    end
end

%% Compile the METIS object files
config.mexflags = {config.mexflags{:} '-DUSE_METIS' ['-I' metispath filesep 'Lib']};
config_metis.mexflags = {config.mexflags{:} '-DUSE_METIS' ['-I' metispath filesep 'Lib']};

% empty strings.h file on windows
if ispc
    fclose(fopen([metispath filesep 'Lib' filesep 'strings.h'], 'w+'));
end

if ~installed
    mexflags = config_metis.mexflags;
    mexflags{end+1} = '-c';
    cd([metispath filesep 'Lib']);
    try
        for i=1:length(LIBMETIS)
            disp(['compiling ' regexprep(LIBMETIS{i}, '\\', '\\\\')]);
            if ~isempty(ver('matlab'))
                % matlab
                mex(mexflags{:}, config_metis.cflags, LIBMETIS{i});
            else
                % octave
                setenv('CFLAGS', config_metis.cflags);
                mex(mexflags{:}, LIBMETIS{i});
                fflush(stdout);
            end
        end
        installed = 1;
    catch
        warning([mfilename ': compilation of METIS failed.']);
        warning(lasterr);
        chdir(curpath);
        return;
    end
    cd(basepath);
end

try
    mexflags = config.mexflags;
    mexflags{end+1} = '-c';
    for i=1:length(SRC)
        fname = SRC{i};
        disp(['compiling ' regexprep(SRC{i}, '\\', '\\\\')]);
        if ~isempty(ver('matlab'))
            % matlab
            mex(mexflags{:}, config.cflags, SRC{i});
        else
            % octave
            setenv('CFLAGS', config.cflags);
            mex(mexflags{:}, SRC{i});
            fflush(stdout);
        end
    end
    
    for i=1:length(MEX_FUNCTIONS)
        fname = MEX_SRC{i};
        disp(['compiling ' regexprep(MEX_SRC{i}, '\\', '\\\\')]);
        ofname = MEX_FUNCTIONS{i};
        if ~isempty(ver('matlab'))
            % matlab
            mex(config.mexflags{:}, config.cflags, MEX_SRC{i}, MEX_OBJ{:}, LIBMETIS_OBJ{:}, config.ldflags, config.mex_output, ofname);
        else
            % octave
            setenv('CFLAGS', config.cflags);
            setenv('LDFLAGS', config.ldflags);
            mex(config.mexflags{:}, MEX_SRC{i}, MEX_OBJ{:}, LIBMETIS_OBJ{:}, config.mex_output, [ofname '.' mexext]);
            fflush(stdout);
        end
    end
    installed = 1;
catch
    warning([mfilename ': compilation of ' fname ' failed.']);
    warning(lasterr);
    installed = 0;
end

chdir(curpath);

end
