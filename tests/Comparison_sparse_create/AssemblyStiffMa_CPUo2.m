function K = AssemblyStiffMa_CPUo2(ELEMS, Ke, opts)
% ASSEMBLYSTIFFMA_CPUO2 Assembly a finite element sparse stiffness matrix K
% with an optimized CPU code.
%
%   K = ASSEMBLYSTIFFMA_CPUO2(ELEMS, Ke, opts) returns a finite element
%   sparse matrix K that is computed with an optimized CPU code
%   (sparse_create) according to the input data, where
%   - "ELEMS": Connectiviy matrix for the Hex8 element mesh (8 x nel)
%   - "Ke": Element stiffness matrix (symmetric part) as a vector (EDOF x (EDOF+1) / 2)
%   - "opts": Options for the sparse_create funtion, in which the fields
%   are required:
%           - opts.symmetric = 1; For symmetric
%           - opts.n_node_dof = ndof; To specify the number of DOFs per
%           node
%
%   See also SPARSE, ACCUMARRAY
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  September 17, 2020. Version: 1.0
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

K = sparse_create(ELEMS, Ke, opts);
