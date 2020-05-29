function Ke = eStiff_vpsa(elements, nodes, MP, sets)
% ESTIFF_VPSA Compute the element stiffness matrices for a VECTOR (v) problem
% using GPU computing (p) taking advantage of simmetry (s) and returning ALL (a)
% ke for the mesh.
%   Ke = ESTIFF_VPSA(elements, nodes, MP, sets) returns the element stiffness
%   matrix "ke" for all elements in a finite element analysis of a vector
%   problem in a three-dimensional domain taking advantage of symmetry and
%   GPU computing, where "elements" is the connectivity matrix, "nodes" the
%   nodal coordinates, and "MP.E" (Young's modulus) and "MP.nu" (Poisson
%   ratio) the material  property for an isotropic material. The struct
%   "sets" must contain several similation parameters:
%   - sets.dTE is the data precision of "Mesh.elements"
%   - sets.dTN is the data precision of "Mesh.nodes"
%   - sets.nel is the number of finite elements
%   - sets.tbs is the Thread Block Size
%   - sets.numSMs is the number of multiprocessors on the device
%   - sets.WarpSize is the warp size
%
%   See also STIFFMA_VPS, ESTIFF_VSS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 09/05/2020. Version: 1.4. Name changed, Doc improved, Support
% 	for only uint32 (removed uint64)
% 	Created: 16/01/2019. Version: 1.0

% MATLAB KERNEL CREATION
if ( strcmp(sets.dTE,'uint32') && strcmp(sets.dTN,'single') )       % Indices: 'uint32'. NNZ: 'single'
    kernel = parallel.gpu.CUDAKernel('eStiff_vpss.ptx', 'eStiff_vpss.cu');
    sets.nel = single(sets.nel);                                    % Converts to 'single' precision
elseif ( strcmp(sets.dTE,'uint32') && strcmp(sets.dTN,'double') )   % Indices: 'uint32'. NNZ: 'double'
    kernel = parallel.gpu.CUDAKernel('eStiff_vpsd.ptx', 'eStiff_vpsd.cu');
else
    error('Input "elements" must be defined as "uint32" and "nodes" as "single" or "double"');
end

% MATLAB KERNEL CONFIGURATION
if (sets.tbs > kernel.MaxThreadsPerBlock || mod(sets.tbs, sets.WarpSize) )
    sets.tbs = kernel.MaxThreadsPerBlock;
    if  mod(sets.tbs, sets.WarpSize)
        sets.tbs = sets.tbs - mod(sets.tbs, sets.WarpSize);
    end
end
kernel.ThreadBlockSize = [sets.tbs, 1, 1];                          % Threads per block
kernel.GridSize = [sets.WarpSize*sets.numSMs, 1, 1];                % Blocks per grid

% INITIALIZATION OF GPU VARIABLES
L = dNdrst(sets.dTN);                                               % Shape functions derivatives in natural coord.
D = DMatrix(MP.E, MP.nu, sets.dTN);                                 % Material matrix (isotropic)

% MATLAB KERNEL CALL
setConstantMemory(kernel,'L',L,'D',D,'nel',sets.nel);               % Set constant memory on GPU
Ke = feval(kernel, elements, nodes, zeros(sets.sz*sets.nel, 1, sets.dTN, 'gpuArray'));% GPU code execution
