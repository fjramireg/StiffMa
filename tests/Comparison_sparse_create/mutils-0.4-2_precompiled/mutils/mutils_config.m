function config = mutils_config(basepath, nowarnings)
%MUTILS_CONFIG sets the compilation flags used to compile mutils
%
%  config = MUTILS_CONFIG(basepath)
%
%Arguments:
%  basepath            : top directory of mutils
%
%Output:
%  config              : structure with configuration information

% Copyright 2012, Marcin Krotkiewski, University of Oslo

config = [];
config.mexflags = {};

if nargin==1
    nowarnings=0;
end

curpath = pwd;
cd(basepath);

config.cflags = [];
config.cxxflags = [];
config.coptimflags = [];
config.ldflags = [];
config.obj_extension = [];

if ~isempty(ver('matlab'))
    config.mex_output = '-output';
else
    config.mex_output = '-o';
end

cc = [];
if exist('mutils_compiler')
    try
        cc = mutils_compiler;
    catch
    end
end

if isempty(cc)
    mex('mutils_compiler.c', config.mex_output, ['mutils_compiler.' mexext]);
end
cc = mutils_compiler;

% compiler
config.mexflags = {config.mexflags{:} ['-I' basepath]};

% turn on/off debuging features
use_debugging=0;
% use_debugging=1;

% turn on/off compile time debug information
use_compiletime_debugging=0;
% use_compiletime_debugging=1;

% turn on/off OpenMP
use_openmp=0;
% use_openmp=1;

% turn on Shewchuk's exact predicates for point in triangle and point in
% tetrahedron tests
use_exact_predicates = 0;
use_exact_predicates = 1;

% metis
metispath = [basepath filesep '..' filesep 'metis' filesep 'sources' filesep 'Lib'];

% compiler flags
if strcmp(cc, 'cl')
    % a flag needed to compile metis
    config.mexflags{end+1} = '-D__VC__';
    if use_debugging
        config.cflags = [config.cflags ' /DDEBUG '];
    end
    if use_compiletime_debugging
        config.cflags = [config.cflags ' /DDEBUG_COMPILE '];
    end
    if use_openmp
        config.cflags = [config.cflags ' /DUSE_OPENMP /openmp '];
        config.ldflags = [config.ldflags ' /openmp'];
    end
    if use_exact_predicates
        config.cflags = [config.cflags ' /DROBUST_PREDICATES '];
    end
    
    %config.cflags = [config.cflags ' /arch:AVX'];
    if exist(metispath)==7
        config.mexflags = {config.mexflags{:} '/DUSE_METIS' ['/I' metispath ] };
    end
end

if strcmp(cc, 'icc')
    config.cflags = [config.cflags ' -std=c99 -fPIC'];
    if nowarnings
        %config.cflags = [config.cflags ' -wformat'];
    else
        config.cflags = [config.cflags ' -Wall'];
    end
    config.cflags = [config.cflags ' -funroll-loops -finline-functions '];
    
    if use_debugging
        config.cflags = [config.cflags ' -DDEBUG '];
    end
    if use_compiletime_debugging
        config.cflags = [config.cflags ' -DDEBUG_COMPILE '];
    end
    if use_openmp
        config.cflags = [config.cflags ' -DUSE_OPENMP -openmp '];
        config.ldflags = [config.ldflags ' -openmp'];
    end
    if use_exact_predicates
        config.cflags = [config.cflags ' -DROBUST_PREDICATES '];
    end
    
    if exist(metispath)==7
        config.mexflags = {config.mexflags{:} '-DUSE_METIS' ['-I' metispath ] };
    end
    
end

if strcmp(cc, 'gcc')
    config.cflags = [config.cflags ' -std=c99 -fPIC'];
    if nowarnings
        config.cflags = [config.cflags ' -Wno-format -Wno-implicit-function-declaration'];
    else
        config.cflags = [config.cflags ' -Wall'];
    end
    
    config.coptimflags = [config.coptimflags ' -fpeephole2 -fschedule-insns2'];
    config.cflags = [config.cflags ' -funroll-loops -finline-functions'];
    
    if use_debugging
        config.cflags = [config.cflags ' -DDEBUG '];
    end
    if use_compiletime_debugging
        config.cflags = [config.cflags ' -DDEBUG_COMPILE '];
    end
    if use_openmp
        config.cflags = [config.cflags ' -DUSE_OPENMP -fopenmp '];
        config.ldflags = [config.ldflags ' -fopenmp'];
    end
    if use_exact_predicates
        config.cflags = [config.cflags ' -DROBUST_PREDICATES '];
    end
    
    if exist(metispath)==7
        config.mexflags = {config.mexflags{:} '-DUSE_METIS' ['-I' metispath ] };
    end
end


% matlab vs octave
if ~isempty(ver('matlab'))
    
    if ispc
        config.cflags = ['COMPFLAGS=$COMPFLAGS ' config.cflags];
        config.cxxflags = ['COMPFLAGS=$COMPFLAGS ' config.cxxflags];
        config.ldflags = ['LINKFLAGS=$LINKFLAGS ' config.ldflags];
        config.coptimflags = 'COPTIMFLAGS=/O2 /DNDEBUG';
        config.obj_extension = '.obj' ;
    else
        config.cflags = ['CFLAGS=\$CFLAGS ' config.cflags];
        config.cxxflags = ['CXXFLAGS=\$CXXFLAGS ' config.cxxflags];
        config.ldflags = ['LDFLAGS=\$LDFLAGS ' config.ldflags];
        config.coptimflags = 'COPTIMFLAGS=-O2 -DNDEBUG';
        config.obj_extension = '.o';
    end
    
    config.mexflags{end+1} = '-O';
    if (~isempty (strfind (computer, '64')))
        config.mexflags{end+1} = '-largeArrayDims';
    end
end
if ~isempty(ver('octave'))
    
    if ispc
        config.cflags = [config.cflags ' /O2 /DNDEBUG /DMATLAB_MEX_FILE'];
        config.obj_extension = '.obj' ;
    else
        config.cflags = [config.cflags ' -O2 -DNDEBUG -DMATLAB_MEX_FILE'];
        config.obj_extension = '.o';
    end
    
end

% check if tcmalloc is available
config.tcmalloc_flags = ' ';
config.tcmalloc_libs = '';
if exist([basepath filesep 'libtcmalloc.a'])
    config.tcmalloc_flags = ' -DUSE_TCMALLOC';
    config.tcmalloc_libs = [' ..' filesep 'libtcmalloc.a'];
end

% mex verbosity for debugging purposes
% config.mexflags{end+1} = '-v';

cd(curpath);

end
