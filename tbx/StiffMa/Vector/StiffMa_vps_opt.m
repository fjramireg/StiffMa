function K = StiffMa_vps_opt(elements, nodes, MP, sets)
% STIFFMA_VPS_OPT Create the global stiffness matrix for a VECTOR (v)
% problem using parallel (p) GPU computing taking advantage of simmetry (s)
% by using some CUDA optimization (opt) techniques.
% 
%   STIFFMA_VPS_OPT(elements,nodes,MP,sets) returns the lower-triangle of
%   a sparse matrix K from finite element analysis of vector problems in a
%   three-dimensional domain taking advantage of simmetry and GPU
%   computing, where "elements" is the connectivity matrix (nel-by-8),
%   "nodes" the nodal coordinates (nnodes-by-3), "MP.E" (Young's modulus)
%   and "MP.nu" (Poisson ratio) the material property for an isotropic
%   material. The struct "sets" must contain several simulation parameters
%   such as:
%   - sets.dTE is the data precision of "elements"
%   - sets.dTN is the data precision of "nodes"
%   - sets.nel is the number of finite elements
%   - sets.sz  is the number of symmetry entries.
%   - sets.tbs is the Thread Block Size
%   - sets.numSMs is the number of multiprocessors on the device
%   - sets.WarpSize is the warp size
%
%   See also SPARSE, ACCUMARRAY, STIFFMA_VSS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Institución Universitaria Pascual Bravo, Medellin-Colombia
%       Created: July 10, 2026. Version: 1.0

%% Inputs check
if ~(existsOnGPU(elements) && existsOnGPU(nodes))                   % Check if "elements" & "nodes" are on GPU memory
    error('Inputs "elements" and "nodes" must be on GPU memory. Use "gpuArray"');
elseif ( size(elements,2)~=8 || size(nodes,2)~=3 )                  % Check if "elements" & "nodes" are nelx8 & nnodx3.
    error('Input "elements" must be a nel-by-8 array, and "nodes" of size nnodes-by-3');
elseif ~( strcmp(sets.dTE,'uint32') || strcmp(sets.dTE,'uint64') )  % Check data type for "elements"
    error('Error. Input "elements" must be "uint32", "uint64" or "double" ');
elseif ~( strcmp(sets.dTN,'single') || strcmp(sets.dTN,'double') )  % Check data type for "nodes"
    error('MATLAB only support "sinlge" sparse matrix for R2025a an after.');
elseif ~( isscalar(MP.E) && isscalar(MP.nu) )                       % Check input "E" and "nu"
    error('Error. Inputs "E" and "nu" must be SCALAR variables');
end

%% Index computation
[iK, jK] = Index_vpsa_opt(elements, sets);      % Row/column indices of tril(K)

%% Element matrix computation
Ke = eStiff_vpsa_opt(elements, nodes, MP, sets);% Entries of tril(K)

%% Assembly of global sparse matrix on GPU
K = AssemblyStiffMa(iK, jK, Ke, sets);	% Global stiffness matrix K assembly
