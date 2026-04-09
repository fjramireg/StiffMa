function K = StiffMa_soCPU2(elements, c, opts)
% STIFFMA_SOCPU2 is a function to generate the STIFFnes MAtrix on the CPU
% for the scalar problem based on optimized CPU code "sparse_create"
% developed in MILAMIN (http://milamin.org/) 
%
% K = STIFFMA_SOCPU2(elements, c, opts) returns the lower-triangle of a
% sparse matrix K from finite element analysis in scalar problems using the
% Hex8 element in a three-dimensional domain taking advantage of symmetry
% and optimized CPU code, where the required inputs are:
%   - "ELEMS": Connectiviy matrix for the Hex8 element mesh (8 x nel)
%   - "c" is the material property (thermal consuctivity)
%   - "opts": Options for the sparse_create funtion, in which the fields
%   are required:
%           - opts.symmetric = 1; For symmetric
%           - opts.n_node_dof = ndof; To specify the number of DOFs per
%           node
% 
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  September 16, 2020.    Version: 1.0
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

%% Numerical integration (only for scalar problem in structured meshes)
Ke = eStiff_sosa2(c);

%% Assembly
K = AssemblyStiffMa_CPUo2(elements', Ke, opts);
