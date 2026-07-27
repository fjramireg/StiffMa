function K = StiffMa_sps_opt(elements, nodes, c, sets)
% STIFFMA_SPS_OPT Create the global stiffness matrix tril(K) for a SCALAR (s)
% problem in PARALLEL (p) GPU computing taking advantage of symmetry (s) using 
% some CUDA optimization (opt) techniques.
% 
%   K = STIFFMA_SPS_OPT(elements,nodes,c,tbs) returns the lower-triangle of a
%   sparse matrix K from finite element analysis of scalar problems in a
%   three-dimensional domain taking advantage of symmetry and GPU
%   computing, where "elements" is the connectivity matrix of size nelx8,
%   "nodes" the nodal coordinates of size Nx3, "c" the material property
%   for an isotropic material (scalar), and the optional "tbs" refers to
%   ThreadBlockSize (scalar). The struct "sets" must contain several
%   similation parameters: 
%   - sets.dTE is the data precision of "Mesh.elements"
%   - sets.dTN is the data precision of "Mesh.nodes"
%   - sets.nel is the number of finite elements
%   - sets.edof is the number of DOFs per element
%   - sets.sz  is the number of symmetry entries
%
%   See also STIFFMA_SPS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Institución Universitaria Pascual Bravo, Medellin-Colombia
%   Created: July 8, 2026. Version: 1.0. Inclusion of some CUDA optimization techniques

%% Inputs check
if ~(existsOnGPU(elements) && existsOnGPU(nodes))	% Check if "elements" & "nodes" are on GPU memory
    error('Inputs "elements" and "nodes" must be on GPU memory. Use "gpuArray"');
elseif ( size(elements,2)~=8 || size(nodes,2)~=3 )	% Check if "elements" & "nodes" are nelx8 & Nx3
    error('Input "elements" must be a nelx8 array, and "nodes" of size Nx3');
elseif ~( strcmp(sets.dTE,'uint32') || strcmp(sets.dTE,'uint64') ) % Check data type for "elements"
    error('Input "elements" must be "uint32" or "uint64"');
elseif ~( strcmp(sets.dTN,'single') || strcmp(sets.dTN,'double') ) % Check data type for "nodes"
    error('MATLAB only support "single" sparse matrix from R2025a');
elseif ~isscalar(c)                                	% Check input "c"
    error('Input "c" must be a SCALAR variable');
end

%% Index computation
[iK, jK] = Index_spsa_opt(elements, sets);  % Row/column indices of tril(K). Size: nel-by-36, 1

%% Element matrix computation
Ke = eStiff_spsa_opt(elements, nodes, c, sets); % Entries of tril(K)

%% Assembly of global sparse matrix on GPU
K = AssemblyStiffMa(iK, jK, Ke, sets);    % Global stiffness matrix K assembly
