function K = StiffMa_vps(elements, nodes, MP, sets)
% STIFFMA_VPS Create the global stiffness matrix for a VECTOR (v) problem
% using parallel (p) GPU computing taking advantage of simmetry (s).
%   STIFFMA_VPS(elements,nodes,MP,sets) returns the lower-triangle of
%   a sparse matrix K from finite element analysis of vector problems in a
%   three-dimensional domain taking advantage of simmetry and GPU
%   computing, where "elements" is the connectivity matrix, "nodes" the
%   nodal coordinates, "MP.E" (Young's modulus) and "MP.nu" (Poisson ratio)
%   the material property for an isotropic material. The struct "sets" must
%   contain several similation parameters:
%   - sets.dTE is the data precision of "elements"
%   - sets.nel is the number of finite elements
%   - sets.sz  is the umber of symmetry entries.
%   - sets.tbs is the Thread Block Size
%   - sets.numSMs is the number of multiprocessors on the device
%   - sets.WarpSize is the warp size
%
%   See also SPARSE, ACCUMARRAY, STIFFMA_VSS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 30/01/2020. Version: 1.4. Name changed, Doc improved
% 	Modified: 28/01/2019. Version: 1.3
%   Created:  16/01/2019. Version: 1.0

%% Inputs check
if ~(existsOnGPU(elements) && existsOnGPU(nodes))                   % Check if "elements" & "nodes" are on GPU memory
    error('Inputs "elements" and "nodes" must be on GPU memory. Use "gpuArray"');
elseif ( size(elements,1)~=8 || size(nodes,1)~=3 )                  % Check if "elements" & "nodes" are 8xnel & 3xnnod.
    error('Input "elements" must be a 8xnel array, and "nodes" of size 3xnnod');
elseif ~( strcmp(sets.dTE,'uint32') || strcmp(sets.dTE,'uint64') )  % Check data type for "elements"
    error('Error. Input "elements" must be "uint32", "uint64" or "double" ');
elseif ~strcmp(sets.dTN,'double')                                  	% Check data type for "nodes"
    error('MATLAB only support "double" sparse matrix, i.e. "nodes" must be of type "double" ');
elseif ~( isscalar(MP.E) && isscalar(MP.nu) )                       % Check input "E" and "nu"
    error('Error. Inputs "E" and "nu" must be SCALAR variables');
end

%% Index computation
[iK, jK] = Index_vpsa(elements, sets);                   % Row/column indices of tril(K)

%% Element matrix computation
Ke = eStiff_vpsa(elements, nodes, MP, sets);            % Entries of tril(K)

%% Assembly of global sparse matrix on GPU
K = AssemblyStiffMa(iK, jK, Ke, sets.dTE, sets.dTN);	% Global stiffness matrix K assembly
