function sz = sizeof( type )
%SIZEOF Get the size in bytes for an element of the specified type
%
%   bytes = sizeof(typename) returns the number of bytes required
%   for a single real value of the specified type.
%
%   Examples:
%   >> sizeof('double')
%   ans = 8
%   >> sizeof('single')
%   ans = 4
%
%   See also: gpuBench

%   Copyright 2011-2014 The MathWorks, Inc.

switch upper(type)
    case {'INT8','UINT8','LOGICAL'}
        sz = 1;
    case {'INT16','UINT16'}
        sz = 2;
    case {'INT32','UINT32'}
        sz = 4;
    case {'INT64','UINT64'}
        sz = 8;
    case 'SINGLE'
        sz = 4;
    case 'DOUBLE'
        sz = 8;
    otherwise
        error( 'SizeOf:BadType', 'Unknown type ''%s''.', type );
end
