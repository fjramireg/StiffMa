function Ke = eStiff_vpsa_opt(elements, nodes, MP, sets)
% ESTIFF_VPSA_OPT Compute the element stiffness matrices for a VECTOR (v)
% problem using parallel GPU computing (p) taking advantage of symmetry (s)
% and returning ALL (a) ke for the mesh.
%
%   Ke = ESTIFF_VPSA_OPT(elements, nodes, MP, sets) returns the element
%   stiffness matrix "ke" for all elements in a finite element analysis of
%   a vector problem in a three-dimensional domain taking advantage of
%   symmetry and GPU computing, where "elements" is the connectivity matrix
%   (nel-by-8), "nodes" the nodal coordinates (nnodes-by-3), and "MP.E"
%   (Young's modulus) and "MP.nu" (Poisson ratio) the material  property
%   for an isotropic material. The struct "sets" must contain several
%   simulation parameters:
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
%   Institución Universitaria Pascual Bravo, Medellin-Colombia
%       Created: July 10, 2026. Version: 1.0

% MATLAB KERNEL CREATION
if ( strcmp(sets.dTE,'uint32') && strcmp(sets.dTN,'single') )      % Indices: 'uint32'. NNZ: 'single'
    kernel = parallel.gpu.CUDAKernel('eStiff_vps_opt.ptx', ...     % PTX code
        'const unsigned int*, const float*, const unsigned int, const unsigned int, float*', ... % C prototype
        'Hex8vector_uint32_single');                               % entry point in the PTX code

elseif ( strcmp(sets.dTE,'uint32') && strcmp(sets.dTN,'double') )   % Indices: 'uint32'. NNZ: 'double'
    kernel = parallel.gpu.CUDAKernel('eStiff_vps_opt.ptx', ...     % PTX code
        'const unsigned int*, const double*, const unsigned int, const unsigned int, double*', ... % C prototype
        'Hex8vector_uint32_double');                               % entry point in the PTX code

elseif ( strcmp(sets.dTE,'uint64') && strcmp(sets.dTN,'double') )   % Indices: 'uint64'. NNZ: 'double'
    kernel = parallel.gpu.CUDAKernel('eStiff_vps_opt.ptx', ...     % PTX code
        'const unsigned long long int*, const double*, const unsigned long long int, const unsigned long long int, double*', ... % C prototype
        'Hex8vector_uint64_double');                               % entry point in the PTX code

else
    msg = sprintf(['Input "elements" must be defined as "uint32" or "uint64", ',...
        'while Input "nodes" must be defined as "single" or "double" when "uint32" is used. ',...
        'However, if "uint64" is defined for "elements", only "double" is accepted for "nodes".']);
    error(msg);
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

% Set constant memory on GPU
if strcmp(sets.dTN,'single')
    setConstantMemory(kernel,'L_single',L,'D_single',D);
elseif strcmp(sets.dTN,'double')
    setConstantMemory(kernel,'L_double',L,'D_double',D);
end

% MATLAB KERNEL CALL
Ke = feval(kernel, ...
    elements, ...
    nodes, ...
    sets.nel, ...
    sets.nnodes, ...
    zeros(sets.sz*sets.nel, 1, sets.dTN, 'gpuArray') ); % GPU code execution
