%% Parallel sparse matrix - vector multiplication
% ex_spmv shows how to
%
% * reorder the sparse matrix to be suited for parallel spmv
% * distribute the matrix among threads using sparse_convert
% * run parallel spmv on native MATLAB matrices
%
%%

function ex_spmv

%% Generate unstructured triangular mesh

% triangle options
opts = [];
opts.max_tri_area = 0.000084;
opts.element_type = 'tri7';
opts.gen_edges = 0;
opts.min_angle = 33;

% domain
tristr.points = [...
    -1 1 1 -1;...
    -1 -1 1 1];
tristr.segments = uint32([...
    1 2 3 4;...
    2 3 4 1]);

% Generate the mesh using triangle
MESH = mtriangle(opts, tristr);


%% Build sparse matrices
% We assemble two sparse matrices - general and symmetric - to test the
% spmv routine. Matrices are assembled for a random symmetric element
% matrix. 

% initialization
ndof   = 3;              % assemble matrices for ndof dofs per node
nnodel = size(MESH.ELEMS, 1);
nel    = size(MESH.ELEMS, 2);
nelemdof = nnodel*ndof;

% generate random symmetric element matrix
Aelem   = rand(nelemdof,nelemdof);
Aelem   = Aelem+Aelem';

% extract lower triangular part from the elemen matrix
Aelem_s = Aelem(find(tril(Aelem)));

%%
% Assemble general sparse matrix using the same element matrix for all elements
opts.n_node_dof = ndof;
opts.symmetric  = 0;
t = tic;
Ag = sparse_create(MESH.ELEMS, Aelem(:), opts);
display(['assemble general sparse matrix (' num2str(ndof) ' dof per node): ' num2str(toc(t))]);
spy(Ag)

%% 
% Assemble symmetric sparse matrix using the same element matrix for all elements
opts.n_node_dof = ndof;
opts.symmetric  = 1;
t = tic;
As = sparse_create(MESH.ELEMS, Aelem_s(:), opts);
display(['assemble symmetric sparse matrix (' num2str(ndof) ' dof per node): ' num2str(toc(t))]);

% compare symmetric and general sparse matrices
if nnz(As-tril(Ag))
    error('symmetric and general matrices created by sparse_create differ');
end


%% Compute communication reducing reordering and permute the matrices
% Parallel spmv requires that the matrix is partitioned among the
% processors. Depending on the way it is partitioned, the cpus 
% exchange different amount of data during computations. For small matrices and
% moderate number of CPUs the Reverse Cuthill-McKee reordering is usually a
% good strategy. For larger number of CPUs and larger systems METIS graph
% partitioning should be used.

% set number of threads
opts.nthreads = 2;

% compute the reordering for a 1 dof per node matrix
Aconn = sparse_create(MESH.ELEMS);

% RCM reordering
t=tic;
perm = mrcm(Aconn);
display(['MRCM:                      ' num2str(toc(t))]);

% METIS graph partitioning
rowdist = [];
if opts.nthreads>1

    % permute with rcm first
    Aconn = Aconn(perm,perm);
    
    t=tic;
    [perm2,~,rowdist] = metis_part(Aconn, opts.nthreads);
    display(['metis_part                 ' num2str(toc(t))]);
    
    % fix the row distribution for multiple dofs per node
    rowdist = uint32((rowdist-1)*ndof+1);

    % merge the permutations
    perm = perm(perm2);
end
clear Aconn;

%%
% The above permutation was created for a symbolic matrix with 1 dof per
% node. In case there are more dofs per node, it needs to be 'expanded'

% block permutation for ndof dofs per node
perm = bsxfun(@plus, -fliplr([0:ndof-1])', ndof*double(perm));
perm = perm(:);

%%
% Now the symmetric and general sparse matrices can be permuted
% using the with the final permutation, which is a combination of RCM and
% METIS graph partitioning.
t=tic;
Ag = cs_permute(Ag,perm,perm);
display(['cs_permute                 ' num2str(toc(t))]);
spy(Ag)

t=tic;
As = cs_symperm(As',perm)';
display(['cs_symperm                 ' num2str(toc(t))]);


%% Parallel sparse matrix - vector multiplication
% spmv MEX function works in two modes:
% * multiply symmetric or general matrix converted by sparse_convert
% * multiply a native MATLAB general matrix
% sparse_convert prepares a native MATLAB sparse matrix for parallel 
% Sparse Matrix Vector Multiplication (SpMV). There are several performance 
% advantages of using this function over simply running SPMV for native 
% MATLAB sparse matrices:
%
%  - uint32 type for row/column indices
%  - symmetric SpMV
%  - interleaved and blocked storage
%  - thread affinity and local memory allocation on NUMA systems
%
% Hence, the first approach is significantly faster if many spmv calls have 
% to be performed and the cost of sparse_convert can be amortized.

% convert a symmetric sparse matrix 
opts.symmetric  = 1;    % symmetric storage
opts.block_size = ndof; % block size to use for Blocked CRS storage
t=tic;
As_converted = sparse_convert(As, opts);
display(['sparse_convert (symmetric, parallel) ' num2str(toc(t))]);

% convert a general sparse matrix
opts.symmetric  = 0;
opts.block_size = ndof;
t=tic;
Ag_converted = sparse_convert(Ag, opts);
display(['sparse_convert (general, parallel)   ' num2str(toc(t))]);


%%
% Run the different versions of spmv and compare the times and
% accuracy of the results

% randomize x vector
x = rand(size(Ag,1), 1);

t = tic;
for i=1:100
    v1 = spmv(As_converted, x);
end
display(['spmv converted, symmetric (parallel)  ' num2str(toc(t))]);

t = tic;
for i=1:100
    v2 = spmv(Ag_converted, x);
end
display(['spmv converted, general (parallel)    ' num2str(toc(t))]);

t = tic;
setenv('OMP_NUM_THREADS', '1');
for i=1:100
    v3 = spmv(Ag, x);
end
display(['spmv native (sequential)   ' num2str(toc(t))]);

t = tic;
setenv('OMP_NUM_THREADS', num2str(opts.nthreads));
for i=1:100
    v3 = spmv(Ag, x);
end
display(['spmv native (parallel)     ' num2str(toc(t))]);

t = tic;
for i=1:100
    v4 = Ag*x;
end
display(['Ag*x                       ' num2str(toc(t))]);
setenv('OMP_NUM_THREADS', '');

%%

% compare the different results
if(norm(v1-v4)>1e-11)
    error('spmv(As_converted,x) and Ag*x returned different results');
end
if(norm(v2-v4)>1e-11)
    error('spmv(Ag_converted,x) and Ag*x returned different results');
end
if(norm(v3-v4)>1e-11)
    error('spmv(Ag,x) and Ag*x returned different results');
end
