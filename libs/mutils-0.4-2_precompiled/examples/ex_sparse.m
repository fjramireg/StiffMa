%% Create sparse matrices for FEM models
% ex_sparse shows how to use sparse_create to
%
% * create a symbolic sparse matrix, i.e., mesh connectivity graph
% * assemble general and symmetric sparse matrices from mesh information
% and element matrices
% * use parallel capabilities of sparse_create
%
% Brief performance comparison between sparse_create, sparse, and sparse2 
% are shown

function ex_sparse

%% Generate unstructured triangular mesh

% Set triangle options
opts = [];
opts.max_tri_area  = 0.00002;
opts.element_type  = 'tri3';
opts.gen_edges     = 0;

% Setup domain - rectangular box
tristr.points   = [-2 2 2 -2; -1 -1 1 1];
tristr.segments = uint32([1 2 3 4; 2 3 4 1]);

% Generate the mesh using triangle
MESH = mtriangle(opts, tristr);


%% Create symbolic sparse matrix using |sparse_create|
% Symbolic sparse matrix has a non-zero entry for every pair of neighboring
% nodes |(i, j)|. It is a logical matrix, in which all non-zero entries are
% equal 1. It is useful e.g., for reordering purposes.

% create general 'symbolic' non-zero connectivity matrix 
% with 1 degree of freedom per node
t = tic;
Ag = sparse_create(MESH.ELEMS);
disp(['general symbolic sparse matrix (1 dof per node):   ' num2str(toc(t))]);

% Verify that Ag is a symmetric sparse matrix
if nnz(Ag-Ag')
    error('Symbolic general matrix created by sparse_create is not symmetric');
end

% create symmetric lower-triangular connectivity matrix
opts.symmetric  = 1;
t = tic;
As = sparse_create(MESH.ELEMS, 1, opts);
disp(['symmetric symbolic sparse matrix (1 dof per node): ' num2str(toc(t))]);

% lower-triangular part of Ag should be the same as As
if nnz(As-tril(Ag))
    error('symmetric and general matrices created by sparse_create differ.');
end

%% Assemble sparse matrix from element matrices
% |sparse_create| can be used to assemble FEM matrices given the element
% list and individual element matrices. This can be done both for symmetric
% and general matrix storage.

nnod     = size(MESH.NODES, 2)  % number of nodes
nel      = size(MESH.ELEMS, 2)  % number of elements
ndof     = 3;                   % number of degrees of freedom per node
nnodel   = size(MESH.ELEMS, 1); % number of nodes per element
nelemdof = nnodel*ndof;

% generate synthetic symmetric element matrix
% the same element matrix is used for all elements
Aelem = rand(nelemdof,nelemdof);
Aelem = Aelem+Aelem';
Aelem = repmat(Aelem(:), 1, nel);

%%

% assemble general sparse matrix
opts.symmetric = 0;
opts.n_node_dof = ndof;
t = tic;
Ag = sparse_create(MESH.ELEMS, Aelem, opts);   
disp(['assemble   general sparse matrix (3 dof per node): ' num2str(toc(t))]);

% extract the lower-triangular part from the element matrices
loidx   = find(tril(true(nnodel*ndof)));
Aelem_s = Aelem(loidx,:);

% assemble symmetric sparse matrix
opts.symmetric = 1;
t = tic;
As = sparse_create(MESH.ELEMS, Aelem_s, opts);   
disp(['assemble symmetric sparse matrix (3 dof per node): ' num2str(toc(t))]);

%%
spy(Ag)

%%
spy(As)

if nnz(As-tril(Ag))
    error('symmetric and general matrices created by sparse_create differ.');
end


%% Parallel assembly of sparse matrices
% For best scalability on even a moderate number of cpus the nodes need to be
% initially reordered using geometric renumbering. Unlike other reorderings, 
% this one only depends on node coordinates, and not on node connectivities.
% Hence, it can be used to improve the performance of |sparse_create|
% without the need to first create the symbolic sparse matrix.
% Note that this ordering also improves the performance in the sequential 
% case due to a better cache reuse.

% compute geometric renumbering and reorder the mesh nodes and elements
[perm,iperm] = geom_order(MESH.NODES);
MESHp = permute_nodes(MESH, perm, iperm);
MESHp = permute_elems(MESHp);

%%

% Create a symmetric sparse matrix based on the renumbered mesh
opts.nthreads = 1;  % sequential execution
t = tic;
As = sparse_create(MESHp.ELEMS, Aelem_s, opts);   
disp(['1 CPU,  assemble renumbered symmetric sparse matrix (3 dof per node): ' num2str(toc(t))]);

% Parallel assembly of symmetric sparse matrix is executed by setting the
% desired number of threads in opts structure.
opts.nthreads = 2;  % parallel execution
t = tic;
As_par = sparse_create(MESHp.ELEMS, Aelem_s, opts);   
disp(['2 CPUs, assemble renumbered symmetric sparse matrix (3 dof per node): ' num2str(toc(t))]);

% compare the results to the sequential version.
if max(max(abs(As_par-As)))>1e-14
    error('sequential and parallel sparse_create gave signifficantly different results.');
end

%%

% Structure of the sparse matrix with nodes renumbered using geometric
% reordering
spy(As_par)
clear As Aelem_s As_perm As_par;


%% Native MATLAB |sparse| function
% To use |sparse| the triplet sparse format has to be prepared. 
% For every element indices of the connectivities between all element 
% degrees of freedom are explicitly enumerated.

% number the element degrees of freedom
t = tic;
ELEM_DOF = zeros(nelemdof, nel);
for dof=1:ndof
    ELEM_DOF(dof:ndof:end,:) = ndof*(MESH.ELEMS-1)+dof;
end

% create connectivities between element degree of freedom
[indx_j indx_i] = meshgrid(1:nnodel*ndof);
A_i = ELEM_DOF(indx_i,:);
A_j = ELEM_DOF(indx_j,:);
disp(['triplet indices: ' num2str(toc(t))]);

%%

% assemble general sparse matrix
t=tic;
A = sparse(A_i, A_j, Aelem);
disp(['assemble general sparse matrix (sparse): ' num2str(toc(t))]);

%%

% compare to the results obtained by sparse_create
if max(max(abs(A-Ag))) > 1e-14
    warning('sparse matrices created by sparse and sparse_create differ too much.');
end
clear Ag Ad;

%% |sparse2| function from SuiteSparse

if exist(['sparse2.' mexext]) == 3
    
    % assemble general sparse matrix
    t=tic;
    Ag = sparse2(A_i, A_j, Aelem);
    disp(['assemble general sparse matrix (sparse2): ' num2str(toc(t))]);
    
    %%
    
    % compare to the results obtained by sparse
    if max(max(abs(A-Ag))) > 1e-14
        warning('sparse matrices created by sparse and sparse2 differ too much.');
    end
end

end % function ex_sparse


%% Auxiliary functions
% Functions used to renumber mesh nodes and elements.

function MESH = permute_nodes(MESH, perm, iperm)

% node list permutation
% element definition permutation
% reshape needed for 1 element - iperm() returns a row-vector
MESH.ELEMS = reshape(iperm(MESH.ELEMS), size(MESH.ELEMS));

if isfield(MESH, 'NODES')
    MESH.NODES = MESH.NODES(:,perm);
end
if isfield(MESH, 'node_markers')
    MESH.node_markers = MESH.node_markers(:,perm);
end

% edges and facets definition permutation
if isfield(MESH, 'EDGES')
    MESH.EDGES = iperm(MESH.EDGES);
end
if isfield(MESH, 'SEGMENTS')
    MESH.SEGMENTS = iperm(MESH.SEGMENTS);
end
if isfield(MESH, 'FACETS')
    MESH.FACETS = iperm(MESH.FACETS);
end
if isfield(MESH, 'FACES')
    MESH.FACES = iperm(MESH.FACES);
end
end


function MESH = permute_elems(MESH)

[~, permel]  = sort(max(MESH.ELEMS));
permel = uint32(permel);

MESH.ELEMS  = MESH.ELEMS(:,permel);

if isfield(MESH, 'elem_markers')
    MESH.elem_markers = MESH.elem_markers(permel);
end

if isfield(MESH, 'ELEMS_EDGES')
    MESH.ELEMS_EDGES = MESH.ELEMS_EDGES(:, permel);
end

if isfield(MESH, 'ELEMS_FACES')
    MESH.ELEMS_FACES = MESH.ELEMS_FACES(:, permel);
end

if isfield(MESH, 'NEIGHBORS')
    % first permute the elements
    MESH.NEIGHBORS = MESH.NEIGHBORS(:, permel);
    
    % now the neighbor information for every element
    noneighbor = (MESH.NEIGHBORS==0);
    MESH.NEIGHBORS(noneighbor) = 1;
    ipermel(permel)= uint32(1:length(permel));
    MESH.NEIGHBORS = ipermel(MESH.NEIGHBORS);
    MESH.NEIGHBORS(noneighbor) = 0;
end
end
