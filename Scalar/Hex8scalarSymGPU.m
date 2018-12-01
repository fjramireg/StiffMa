%  * ====================================================================*/
% ** This function was developed by:
%  *          Francisco Javier Ramirez-Gil
%  *          Universidad Nacional de Colombia - Medellin
%  *          Department of Mechanical Engineering
%  *
%  ** Please cite this code as:
%  *
%  ** Date & version
%  *      01/12/2018.
%  *      V 1.3
%  *
%  * ====================================================================*/

function KE = Hex8scalarSymGPU(elements,nodes,c)
% NNZ values of symmetric part of the element stiffness matrix ke (SCALAR)

% Inputs check
if ~(existsOnGPU(elements) && existsOnGPU(nodes))   % Check if "elements" and "nodes" are on GPU memory
    error('Error. Input "elements" must be a gpuArray');
elseif size(elements,1) ~= 8                        % Check if "elements" is an array of size 8xnel
    error('Error. Input "elements" must be a 8xnel array');
elseif size(nodes,1) ~= 3                           % Check if "nodes" is an array of size 3xnnod
    error('Error. Input "nodes" must be a 3xnnod array');
elseif ~( strcmp(classUnderlying(elements),'uint32') || strcmp(classUnderlying(elements),'uint64') )
    error('Error. Input "elements" must be "uint32" or "uint64"');
elseif ~( strcmp(classUnderlying(nodes),'single') || strcmp(classUnderlying(nodes),'double') )
    error('Error. Input "nodes" must be "single" or "double"');
elseif ~isscalar(c)
    error('Error. Input "c" must be a SCALAR variable');
end

% General variables
nel = size(elements,2);                              % Number of elements
nnod = size(nodes,2);                                % Number of nodes
L = dNdrst;                                          % Shape functions derivatives in natural coord.

% Indices of type 'uint32' and NNZ values of type 'single'
if ( strcmp(classUnderlying(elements),'uint32') && strcmp(classUnderlying(nodes),'single') )
    ker = parallel.gpu.CUDAKernel('Hex8scalarSymGPU.ptx',...    % PTXFILE
        'const unsigned int *, const float *, float *',...      % C prototype for kernel
        'Hex8scalarIfj');                                       % Specify entry point
    ker.ThreadBlockSize = [ker.MaxThreadsPerBlock, 1, 1];       % Threads per block
    ker.GridSize = [ceil(nel/ker.ThreadBlockSize(1)), 1, 1];    % Blocks per grid
    setConstantMemory(ker,'L',L,'nel',nel,'nnod',nnod,'c',c);   % Set constant memory on GPU
    KE = feval(ker, elements, nodes, zeros(36*nel,1,'single','gpuArray')); % GPU code execution
    
    % Indices of type 'uint32' and NNZ values of type 'double'
elseif ( strcmp(classUnderlying(elements),'uint32') && strcmp(classUnderlying(nodes),'double') )
    ker = parallel.gpu.CUDAKernel('Hex8scalarSymGPU.ptx',...    % PTXFILE
        'const unsigned int *, const double *, double *',...    % C prototype for kernel
        'Hex8scalarIdj');                                       % Specify entry point
    ker.ThreadBlockSize = [ker.MaxThreadsPerBlock, 1, 1];       % Threads per block
    ker.GridSize = [ceil(nel/ker.ThreadBlockSize(1)), 1, 1];    % Blocks per grid
    setConstantMemory(ker,'L',L,'nel',nel,'nnod',nnod,'c',c);   % Set constant memory on GPU
    KE = feval(ker, elements, nodes, zeros(36*nel,1,'double','gpuArray')); % GPU code execution
    
    % Indices of type 'uint64' and NNZ values of type 'single'
elseif ( strcmp(classUnderlying(elements),'uint64') && strcmp(classUnderlying(nodes),'single') )
    ker = parallel.gpu.CUDAKernel('Hex8scalarSymGPU.ptx',...    % PTXFILE
        'const unsigned long *, const float *, float *',...     % C prototype for kernel
        'Hex8scalarIfm');                                       % Specify entry point
    ker.ThreadBlockSize = [ker.MaxThreadsPerBlock, 1, 1];       % Threads per block
    ker.GridSize = [ceil(nel/ker.ThreadBlockSize(1)), 1, 1];    % Blocks per grid
    setConstantMemory(ker,'L',L,'nel',nel,'nnod',nnod,'c',c);   % Set constant memory on GPU
    KE = feval(ker, elements, nodes, zeros(36*nel,1,'single','gpuArray')); % GPU code execution
    
    % Indices of type 'uint64' and NNZ values of type 'double'
elseif ( strcmp(classUnderlying(elements),'uint64') && strcmp(classUnderlying(nodes),'double') )
    ker = parallel.gpu.CUDAKernel('Hex8scalarSymGPU.ptx',...    % PTXFILE
        'const unsigned long *, const double *, double *',...   % C prototype for kernel
        'Hex8scalarIdm');                                       % Specify entry point
    ker.ThreadBlockSize = [ker.MaxThreadsPerBlock, 1, 1];       % Threads per block
    ker.GridSize = [ceil(nel/ker.ThreadBlockSize(1)), 1, 1];    % Blocks per grid
    setConstantMemory(ker,'L',L,'nel',nel,'nnod',nnod,'c',c);   % Set constant memory on GPU
    KE = feval(ker, elements, nodes, zeros(36*nel,1,'double','gpuArray')); % GPU code execution
end
