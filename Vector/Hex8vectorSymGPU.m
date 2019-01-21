%  * ====================================================================*/
% ** This function was developed by:
%  *          Francisco Javier Ramirez-Gil
%  *          Universidad Nacional de Colombia - Medellin
%  *          Department of Mechanical Engineering
%  *
%  ** Please cite this code as:
%  *
%  ** Date & version
%  *      Created: 17/01/2019. Last modified: 21/01/2019
%  *      V 1.3
%  *
%  * ====================================================================*/

function KE = Hex8vectorSymGPU(elements,nodes,E,nu)
% Symmetric part of the element stiffness matrix ke (VECTOR-DOUBLE)

% General variables
dTypeE = classUnderlying(elements);     % Data precision of "elements"
dTypeN = classUnderlying(nodes);        % Data precision of "nodes"
nel  = size(elements,2);                % Number of elements
nnod = size(nodes,2);                   % Number of nodes
L = dNdrst(dTypeN);                     % Shape functions derivatives in natural coord.
D = MaterialMatrix(E,nu,dTypeN);        % Material matrix (isotropic)

% Check the data type to create the proper CUDA kernel object
if ( strcmp(dTypeE,'int32') && strcmp(dTypeN,'single') )        % Indices: 'int32'. NNZ: 'single'
    ker = parallel.gpu.CUDAKernel('Hex8vectorSymGPU.ptx',...    % PTXFILE
        'const int *, const float *, float *',...               % C prototype for kernel
        'Hex8vectorIfi');                                       % Specify entry point
elseif ( strcmp(dTypeE,'uint32') && strcmp(dTypeN,'single') )   % Indices: 'uint32'. NNZ: 'single'
    ker = parallel.gpu.CUDAKernel('Hex8vectorSymGPU.ptx',...
        'const unsigned int *, const float *, float *',...
        'Hex8vectorIfj');
elseif ( strcmp(dTypeE,'int32') && strcmp(dTypeN,'double') )    % Indices: 'int32'. NNZ: 'double'
    ker = parallel.gpu.CUDAKernel('Hex8vectorSymGPU.ptx',...
        'const int *, const double *, double *',...
        'Hex8vectorIdi');
elseif ( strcmp(dTypeE,'uint32') && strcmp(dTypeN,'double') )   % Indices: 'uint32'. NNZ: 'double'
    ker = parallel.gpu.CUDAKernel('Hex8vectorSymGPU.ptx',...
        'const unsigned int *, const double *, double *',...
        'Hex8vectorIdj');
elseif ( strcmp(dTypeE,'int64') && strcmp(dTypeN,'double') )    % Indices: 'int64'. NNZ: 'double'
    ker = parallel.gpu.CUDAKernel('Hex8vectorSymGPU.ptx',...
        'const long *, const double *, double *',...
        'Hex8vectorIdl');
elseif ( strcmp(dTypeE,'uint64') && strcmp(dTypeN,'double') )   % Indices: 'uint64'. NNZ: 'double'
    ker = parallel.gpu.CUDAKernel('Hex8vectorSymGPU.ptx',...
        'const unsigned long *, const double *, double *',...
        'Hex8vectorIdm');
elseif ( strcmp(dTypeE,'double') && strcmp(dTypeN,'double') )    % Indices: 'double'. NNZ: 'double'
    ker = parallel.gpu.CUDAKernel('Hex8vectorSymGPU.ptx',...
        'const double *, const double *, double *',...
        'Hex8vectorIdd');
else
    error('Input "elements" must be defined as "int32", "uint32", "int64", "uint64" or "double" and "nodes" as "single" or "double"');
end

% Configures and executes the CUDA kernel
ker.ThreadBlockSize = [ker.MaxThreadsPerBlock, 1, 1];               % Threads per block
ker.GridSize = [ceil(nel/ker.ThreadBlockSize(1)), 1, 1];            % Blocks per grid
setConstantMemory(ker,'L',L,'D',D,'nel',nel,'nnod',nnod);           % Set constant memory on GPU
KE = feval(ker, elements, nodes, zeros(300*nel,1,dTypeN,'gpuArray'));% GPU code execution
