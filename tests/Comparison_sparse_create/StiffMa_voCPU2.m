function K = StiffMa_voCPU2(elements, MP, opts)
% STIFFMA_VOCPU2 is a function to generate the STIFFnes MAtrix on the CPU
% for the vector problem based on optimized CPU code "sparse_create"
% developed in MILAMIN (http://milamin.org/)
%
% K = STIFFMA_VOCPU2(elements, MP, opts) returns the lower-triangle of a
% sparse matrix K from finite element analysis in vector problems using the
% Hex8 element in a three-dimensional domain taking advantage of symmetry
% and optimized CPU code, where the required inputs are:
%   - "ELEMS": Connectiviy matrix for the Hex8 element mesh (8 x nel)
%   - "MP" is the material property
%       - "MP.E" is the Young's modulus
%       - "MP.nu" is the Poisson ratio
%   - "opts": Options for the sparse_create funtion, in which the fields
%   are required:
%      	- opts.symmetric = 1; For symmetric
%      	- opts.n_node_dof = ndof; To specify the number of DOFs per node
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  September 17, 2020.    Version: 1.0
%
% Credits:
% The original MILAMIN paper: Dabrowski, M., M. Krotkiewski, and D. W.
% Schmid. MILAMIN: MATLAB-based finite element method solver for large
% problems, Geochem. Geophys. Geosyst., 9, Q04030, 2008.
% https://doi.org/10.1029/2007GC001719
%
% The MUTILS package: Krotkiewski, M. and M. Dabrowski. Parallel symmetric
% sparse matrix-vector product on scalar multi-core cpus. Parallel
% Computing, 36(4):181–198,  2010.
% https://doi.org/10.1016/j.parco.2010.02.003
%

%% Indices computation
% Indices are not required when using sparse_create

%% Numerical integration (only for vector problem in structured meshes)
Ke = eStiff_vosa2(MP);

%% Assembly
K = AssemblyStiffMa_CPUo2(elements', Ke, opts);
