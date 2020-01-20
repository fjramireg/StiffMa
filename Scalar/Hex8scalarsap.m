function KE = Hex8scalarsap(elements,nodes,c,dTE,dTN,tbs)
% HEX8SCALARSAP Compute all tril(ke) for a SCALAR problem in PARALLEL computing
% taking advantage of simmetry and GPU computing.
%   HEX8SCALARSAP(elements,nodes,c,dTE,dTN,tbs) returns the element stiffness
%   matrix "ke" for all elements in a finite element analysis of a scalar
%   problem in a three-dimensional domain taking advantage of symmetry and GPU
%   computing, where "elements" is the connectivity matrix of size 8xnel,
%   "nodes" the nodal coordinates of size 3xN, "c" the material property for an
%   isotropic material (scalar), and "tbs" in an optional input referred to
%   ThreadBlockSize (scalar). dTE and dTN are the data type for the connectiviyt
%   array nodal coordinates array, respectively.
%
%   See also STIFFMAPS, HEX8SCALARSAS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Modified: 04/12/2019. Version: 1.4. Variable number of inputs, Name changed, Doc improved
%   Modified: 21/01/2019. Version: 1.3
%   Created:  01/12/2018. Version: 1.4

% General variables
nel = size(elements,2);                 % Number of elements
L = dNdrst(dTN);                        % Shape functions derivatives in natural coord.

% Check the data type to create the proper CUDA kernel object
% NNZ of type 'single' and indices 'uint32'
if ( strcmp(dTE,'uint32') && strcmp(dTN,'single') )       % Indices: 'uint32'. NNZ: 'single'
    ker = parallel.gpu.CUDAKernel('Hex8scalarsp.ptx',...       % PTXFILE
        'const unsigned int *, const float *, float *',...      % C prototype for kernel
        'Hex8scalarIfj');                                       % Specify entry point
    nel = single(nel);                                          % Converts to 'single' precision
    c   = single(c);
    % NNZ of type 'double' and indices 'uint32', 'uint64', and 'double'
elseif ( strcmp(dTE,'uint32') && strcmp(dTN,'double') )   % Indices: 'uint32'. NNZ: 'double'
    ker = parallel.gpu.CUDAKernel('Hex8scalarsp.ptx',...
        'const unsigned int *, const double *, double *',...
        'Hex8scalarIdj');
elseif ( strcmp(dTE,'uint64') && strcmp(dTN,'double') )   % Indices: 'uint64'. NNZ: 'double'
    ker = parallel.gpu.CUDAKernel('Hex8scalarsp.ptx',...
        'const unsigned long *, const double *, double *',...
        'Hex8scalarIdm');
elseif ( strcmp(dTE,'double') && strcmp(dTN,'double') )   % Indices: 'double'. NNZ: 'double'
    ker = parallel.gpu.CUDAKernel('Hex8scalarsp.ptx',...
        'const double *, const double *, double *',...
        'Hex8scalarIdd');
else
    error('Input "elements" must be defined as "uint32", "uint64" or "double" and Input "nodes" must be defined as "single" or "double" ');
end

% Configure and execute the CUDA kernel
if (nargin == 3 || tbs > ker.MaxThreadsPerBlock)
    tbs = ker.MaxThreadsPerBlock;                                   % Default (MaxThreadsPerBlock)
end
ker.ThreadBlockSize = [tbs, 1, 1];                                  % Threads per block
ker.GridSize = [ceil(nel/ker.ThreadBlockSize(1)), 1, 1];            % Blocks per grid
setConstantMemory(ker,'L',L,'nel',nel,'c',c);                       % Set constant memory on GPU
KE = feval(ker, elements, nodes, zeros(36*nel,1,dTN,'gpuArray'));% GPU code execution
