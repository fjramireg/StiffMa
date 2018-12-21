%  * ====================================================================*/
% ** This function was developed by:
%  *          Francisco Javier Ramirez-Gil
%  *          Universidad Nacional de Colombia - Medellin
%  *          Department of Mechanical Engineering
%  *
%  ** Please cite this code as:
%  *
%  ** Date & version
%  *      13/12/2018.
%  *      V 1.2
%  *
%  * ====================================================================*/

function K = AssemblyScalarSymGPU(elements,nodes,c)
% Construction of the global stiffness matrix K (SCALAR-SYMMETRIC)

%% General declarations
dTE = classUnderlying(elements);            % "elements" data precision. Defines data type of [iK,jK]
dTN = classUnderlying(nodes);              	% "nodes" data precision. Defines data type of [Ke]
N = size(nodes,2);                          % Total number of nodes (DOFs)

%% Inputs check
if ~(existsOnGPU(elements) && existsOnGPU(nodes)) % Check if "elements" & "nodes" are on GPU memory
    error('Inputs "elements" and "nodes" must be on GPU memory. Use "gpuArray"');
elseif ( size(elements,1)~=8 || size(nodes,1)~=3 )% Check if "elements" & "nodes" are 8xnel & 3xnnod.
    error('Input "elements" must be a 8xnel array, and "nodes" of size 3xnnod');
elseif ~( strcmp(dTE,'int32') || strcmp(dTE,'uint32')... % Check data type for "elements"
        || strcmp(dTE,'int64')  || strcmp(dTE,'uint64') || strcmp(dTE,'double') )
    error('Error. Input "elements" must be "int32", "uint32", "int64", "uint64" or "double" ');
elseif ~strcmp(dTN,'double')                      % Check data type for "nodes"
    error('MATLAB only support "double" sparse matrix, i.e. "nodes" must be of type "double" ');
elseif ~isscalar(c)                               % Check input "c"
    error('Error. Input "c" must be a SCALAR variable');
end

%% Index computation
[iK, jK] = IndexScalarSymGPU(elements);     % Row/column indices of tril(K)

%% Element matrix computation
Ke = Hex8scalarSymGPU(elements,nodes,c);   	% Entries of tril(K)

%% Assembly of global sparse matrix
if ( strcmp(dTE,'double') && strcmp(dTN,'double') )
    K = sparse(iK, jK, Ke, N, N);           % Assembly of K on GPU
else
    K = accumarray([iK,jK], Ke, [N,N], [], [], 1);
end
