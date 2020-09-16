function config = compiler_flags
%COMPILER_FLAGS returns platform-dependent structure with compiler settings

% Copyright 2012, Marcin Krotkiewski, University of Oslo

config = [];
v = ver;

if ~isempty(ver('matlab'))
    config.mexflags = '-largeArrayDims';
    config.mex_output = '-output';
else
    config.mex_output = '-o';
end

if ispc
    config.cflags = 'COMPFLAGS=$COMPFLAGS';
    config.cxxflags = 'COMPFLAGS=$COMPFLAGS';
    config.ldflags = 'LINKFLAGS=$LINKFLAGS';
    config.obj_extension = '.obj' ;
else
    config.cflags = 'CFLAGS=\$CFLAGS';
    config.cxxflags = 'CXXFLAGS=\$CXXFLAGS';
    config.ldflags = 'LDFLAGS=\$LDFLAGS';
    config.obj_extension = '.o' ;
end
end
