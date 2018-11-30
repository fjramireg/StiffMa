%  * ====================================================================*/
% ** This function was developed by:
%  *          Francisco Javier Ramirez-Gil
%  *          Universidad Nacional de Colombia - Medellin
%  *          Department of Mechanical Engineering
%  *
%  ** Please cite this code as:
%  *
%  ** Date & version
%  *      30/11/2018.
%  *      V 1.2
%  *
%  * ====================================================================*/
 
function [iK, jK] = IndexScalarSymGPU(elements)
% Row/column indices of the lower triangular part of the sparse stiffness matrix K (SCALAR)

% Inputs check
if ~existsOnGPU(elements)                       % Check if "elements" is already on GPU memory
    error('Error. Input "elements" must be a gpuArray');
elseif size(elements,1) ~= 8                    % Check if "elements" is an array of size 8xnel
    error('Error. Input "elements" must be a 8xnel array');
elseif ~( strcmp(classUnderlying(elements),'uint32') || strcmp(classUnderlying(elements),'uint64') )
    error('Error. Input "elements" must be "uint32" or "uint64"');
end

% Indices of type 'uint32'
if strcmp(classUnderlying(elements),'uint32')
    % INITIALIZATION OF GPU VARIABLES
    nel = size(elements,1);                     % Number of elements 
    iK  = zeros(36*nel,1,'uint32','gpuArray');  % Stores row indices (initialized directly on GPU)
    jK  = zeros(36*nel,1,'uint32','gpuArray');  % Stores column indices (initialized directly on GPU)
    
    % MATLAB KERNEL CREATION
    ker_uint32 = parallel.gpu.CUDAKernel('IndexScalarGPU.ptx',...
        'const unsigned int *, const unsigned int, unsigned int *, unsigned int *','IndexScalarGPUIj');
    
    % MATLAB KERNEL CONFIGURATION
    ker_uint32.ThreadBlockSize = [512, 1, 1];                              % Threads per block
    ker_uint32.GridSize = [ceil(nel/ker_uint32.ThreadBlockSize(1)), 1, 1]; % Blocks per grid
    
    % MATLAB KERNEL CALL
    [iK, jK] = feval(ker_uint32, elements, nel, iK, jK);      % GPU code execution
    
% Indices of type 'uint64'
elseif strcmp(classUnderlying(elements),'uint64')
    % INITIALIZATION OF GPU VARIABLES
    nel = size(elements,1);                     % Number of elements 
    iK  = zeros(36*nel,1,'uint64','gpuArray');  % Stores row indices (initialized directly on GPU)
    jK  = zeros(36*nel,1,'uint64','gpuArray');  % Stores column indices (initialized directly on GPU)
    
    % MATLAB KERNEL CREATION
    ker_uint64 = parallel.gpu.CUDAKernel('IndexScalarGPU.ptx',...
        'const unsigned long *, const unsigned long, unsigned long *, unsigned long *','IndexScalarGPUIm');
    
    % MATLAB KERNEL CONFIGURATION
    ker_uint64.ThreadBlockSize = [512, 1, 1];                              % Threads per block
    ker_uint64.GridSize = [ceil(nel/ker_uint64.ThreadBlockSize(1)), 1, 1]; % Blocks per grid
    
    % MATLAB KERNEL CALL
    [iK, jK] = feval(ker_uint64, elements, nel, iK, jK);                   % GPU code execution
end
