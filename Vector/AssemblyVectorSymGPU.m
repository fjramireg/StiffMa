%  * ====================================================================*/
% ** This function was developed by:
%  *          Francisco Javier Ramirez-Gil
%  *          Universidad Nacional de Colombia - Medellin
%  *          Department of Mechanical Engineering
%  *
%  ** Please cite this code as:
%  *
%  ** Date & version
%  *      17/01/2019.
%  *      V 1.2
%  *
%  * ====================================================================*/

function K = AssemblyVectorSymGPU(elements,nodes,E,nu)
% Construction of the global stiffness matrix K (VECTOR-SYMMETRIC-GPU)

%% General declarations
dTE = classUnderlying(elements);            % "elements" data precision. Defines data type of [iK,jK]
dTN = classUnderlying(nodes);              	% "nodes" data precision. Defines data type of [Ke]
N = size(nodes,2);                          % Total number of nodes

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
elseif ~( isscalar(E) && isscalar(nu) )           % Check input "E" and "nu"
    error('Error. Inputs "E" and "nu" must be SCALAR variables');
end

%% Index computation
[iK, jK] = IndexVectorSymGPU(elements);             % Row/column indices of tril(K)

%% Element matrix computation
Ke = Hex8vectorSymGPU(elements,nodes,E,nu);         % Entries of tril(K)

%% Assembly of global sparse matrix on GPU
if ( strcmp(dTE,'double') && strcmp(dTN,'double') )
    K = sparse(iK, jK, Ke, 3*N, 3*N);
else
    K = accumarray([iK,jK], Ke, [3*N,3*N], [], [], 1);
end
