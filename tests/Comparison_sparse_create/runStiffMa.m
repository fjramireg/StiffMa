
nel = 10;                  % Number of elements at each direction
sets.sf = 1;                % Safety factor. Positive integer to add more partitions
sets.prob_type = 'Vector';  % 'Scalar' or 'Vector'
sets.dTE = 'uint32';        % Data precision for "elements"
sets.dTN = 'double';        % Data precision for "nodes"

% Material properties
c = 384.1;
MP.E = 200e9;
MP.nu = 0.3;

% Add path
addpath(genpath('../../libs/mutils-0.4-2'));
addpath(genpath('../../tbx/StiffMa'));

% Mesh creation
[elements, ~] = CreateMesh2(nel, nel, nel, sets);

% Options for sparse_create
if strcmp(sets.prob_type, 'Scalar')
    ndof = 1;
elseif strcmp(sets.prob_type, 'Vector')
    ndof = 3;
end
opts.symmetric = 1;         % Symmetry
opts.n_node_dof = ndof;     % DOFs per node
% nt = maxNumCompThreads;
% opts.nthreads = nt;     % parallel execution
% % Parallel assembly of sparse matrices
% % For best scalability on even a moderate number of cpus the nodes need to be
% % initially reordered using geometric renumbering. Unlike other reorderings,
% % this one only depends on node coordinates, and not on node connectivities.
% % Hence, it can be used to improve the performance of |sparse_create|
% % without the need to first create the symbolic sparse matrix.
% % Note that this ordering also improves the performance in the sequential
% % case due to a better cache reuse.

% Execute the
if strcmp(sets.prob_type, 'Scalar')
    K = StiffMa_soCPU2(elements, c, opts);
elseif strcmp(sets.prob_type, 'Vector')
    K = StiffMa_voCPU2(elements, MP, opts);
end
