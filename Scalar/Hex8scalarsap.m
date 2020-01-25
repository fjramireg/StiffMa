function KE = Hex8scalarsap(elements, nodes, c, settings)
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
nel = settings.nel;         % Number of elements
L   = dNdrst(settings.dTN);	% Shape functions derivatives in natural coord.

% MATLAB KERNEL CREATION
if ( strcmp(settings.dTE,'uint32') && strcmp(settings.dTN,'single') )       % Indices: 'uint32'. NNZ: 'single'
    ker = parallel.gpu.CUDAKernel('Hex8scalarsps.ptx',...	% PTXFILE
        'const unsigned int *, const float *, float *',...	% C prototype for kernel
        'Hex8scalarIfj');                                 	% Specify entry point
    nel = single(nel);                                     	% Converts to 'single' precision
    c   = single(c);
elseif ( strcmp(settings.dTE,'uint32') && strcmp(settings.dTN,'double') )	% Indices: 'uint32'. NNZ: 'double'
    ker = parallel.gpu.CUDAKernel('Hex8scalarsp.ptx',...
        'const unsigned int *, const double *, double *',...
        'Hex8scalarIdj');
elseif ( strcmp(settings.dTE,'uint64') && strcmp(settings.dTN,'double') )	% Indices: 'uint64'. NNZ: 'double'
    ker = parallel.gpu.CUDAKernel('Hex8scalarsp.ptx',...
        'const unsigned long *, const double *, double *',...
        'Hex8scalarIdm');
else
    error('Input "elements" must be defined as "uint32", "uint64" and Input "nodes" must be defined as "single" or "double" ');
end

% MATLAB KERNEL CONFIGURATION
if (settings.tbs > ker.MaxThreadsPerBlock || mod(settings.tbs, settings.WarpSize) )
    settings.tbs = ker.MaxThreadsPerBlock;
    if  mod(settings.tbs, settings.WarpSize)
        settings.tbs = settings.tbs - mod(settings.tbs, settings.WarpSize);
    end
end
ker.ThreadBlockSize = [settings.tbs, 1, 1];               	% Threads per block
ker.GridSize = [settings.WarpSize*settings.numSMs, 1, 1];	% Blocks per grid   

% INITIALIZATION OF GPU VARIABLES
setConstantMemory(ker, 'L',L, 'nel',nel, 'c',c);            % Set constant memory on GPU
KE = zeros(36*nel, 1, settings.dTN, 'gpuArray');            % Initialized directly on GPU
KE = feval(ker, elements, nodes, KE);                       % GPU code execution
