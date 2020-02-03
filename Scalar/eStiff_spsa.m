function KE = eStiff_spsa(elements, nodes, c, sets)
% ESTIFF_SPSA Computes the element stiffness matrices for a SCALAR (s)
% problem in PARALLEL (P) GPU computing taking advantage of symmetry (s)
% and returning ALL (a) ke for the mesh.
%   ESTIFF_SPSA(elements,nodes,c,sets) returns the element stiffness matrix
%   "ke" for all elements in a finite element analysis of a scalar problem
%   in a three-dimensional domain taking advantage of symmetry and GPU
%   computing, where "elements" is the connectivity matrix of size 8xnel,
%   "nodes" the nodal coordinates of size 3xN, "c" (conductivity) is the
%   material property for an isotropic material (scalar). The struct "sets"
%   must contain several similation parameters:
%   - sets.dTE is the data precision of "Mesh.elements"
%   - sets.dTN is the data precision of "Mesh.nodes"
%   - sets.nel is the number of finite elements
%   - sets.sz  is the number of symmetry entries
%   - sets.tbs is the Thread Block Size
%   - sets.numSMs is the number of multiprocessors on the device
%   - sets.WarpSize is the warp size
%
%   See also STIFFMA_SPS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Modified: 04/12/2019. Version: 1.4. Variable number of inputs, Name changed, Doc improved
%   Modified: 21/01/2019. Version: 1.3
%   Created:  01/12/2018. Version: 1.4

% MATLAB KERNEL CREATION
if ( strcmp(sets.dTE,'uint32') && strcmp(sets.dTN,'single') )       % Indices: 'uint32'. NNZ: 'single'
    ker = parallel.gpu.CUDAKernel('eStiff_spss.ptx',...     % PTXFILE
        'const unsigned int *, const float *, float *',...	% C prototype for kernel
        'Hex8scalarIfj');                                 	% Specify entry point
    sets.nel = single(sets.nel);                            % Converts to 'single' precision
    c = single(c);
elseif ( strcmp(sets.dTE,'uint32') && strcmp(sets.dTN,'double') )	% Indices: 'uint32'. NNZ: 'double'
    ker = parallel.gpu.CUDAKernel('eStiff_sps.ptx',...
        'const unsigned int *, const double *, double *',...
        'Hex8scalarIdj');
elseif ( strcmp(sets.dTE,'uint64') && strcmp(sets.dTN,'double') )	% Indices: 'uint64'. NNZ: 'double'
    ker = parallel.gpu.CUDAKernel('eStiff_sps.ptx',...
        'const unsigned long *, const double *, double *',...
        'Hex8scalarIdm');
else
    error('Input "elements" must be defined as "uint32", "uint64" and Input "nodes" must be defined as "single" or "double" ');
end

% MATLAB KERNEL CONFIGURATION
if (sets.tbs > ker.MaxThreadsPerBlock || mod(sets.tbs, sets.WarpSize) )
    sets.tbs = ker.MaxThreadsPerBlock;
    if  mod(sets.tbs, sets.WarpSize)
        sets.tbs = sets.tbs - mod(sets.tbs, sets.WarpSize);
    end
end
ker.ThreadBlockSize = [sets.tbs, 1, 1];               	% Threads per block
ker.GridSize = [sets.WarpSize*sets.numSMs, 1, 1];       % Blocks per grid

% INITIALIZATION OF GPU VARIABLES
L = dNdrst(sets.dTN);               % Shape functions derivatives in natural coord.
setConstantMemory(ker, 'L',L, 'nel',sets.nel, 'c',c);  	% Set constant memory on GPU

% MATLAB KERNEL CALL
KE = feval(ker, elements, nodes, zeros(sets.sz*sets.nel, 1, sets.dTN, 'gpuArray'));
