function Ac = sparse_convert(varargin)
%SPARSE_CONVERT prepares a native MATLAB sparse matrix for parallel SpMV
%
%  Ac = SPARSE_CONVERT(spA, [opts], [rowdist])
%
%This function prepares a native MATLAB sparse matrix for parallel 
%Sparse Matrix Vector Multiplication (SpMV). There are several performance 
%advantages of using this function over simply running SPMV for native 
%MATLAB sparse matrices:
%
%  - uint32 type for row/column indices
%  - symmetric SpMV
%  - interleaved and blocked storage
%  - thread affinity and local memory allocation on NUMA systems
%
%Arguments:
%  spA            : sparse matrix, symmetric 
%  opts           : structure containing the following fields:
%
%    opts.remove_zero_cols[=0] remove empty columns from local thread matrices
%    opts.blocksize[=1]     size of natural square blocks in the matrix structure 
%    opts.interleave[=0]    use interleaved CRS storage
%    opts.symmetric[=0]     treat the input matrix as a general matrix (0),
%                           or lower-triangular part of a symmetric matrix (1).
%    opts.nthreads[=0]      number of worker threads for the parallel
%                           part of the code. 0 means the value is taken
%                           from the environment variable OMP_NUM_THREADS
%    opts.verbosity[=0]     display some internal timing information
%    opts.cpu_affinity[=0]  use thread to CPU binding for parallel execution
%    opts.cpu_start[=0]     bind first thread to CPU cpu_start
%
%  rowdist        : row distribution of matrix among the threads. rowdist
%                   is a vector of size [1 x nthreads+1]. Entry i holds
%                   the starting row number for thread i. The last entry in
%                   rowdist is the dimension of the matrix + 1.
%
%Output:
%  Ac             : a structure containing converted sparse matrix that can
%                   be used with the spmv function
%
%There are several things done here:
%
%  - if rowdist is not passed as parameter, it is computed so that the
%    amount of work per thread (matrix non-zero entries) is balanced
%  - threads are started using OpenMP
%  - on architectures that support it, threads are bound to their own
%    processors/cores
%  - local matrix parts are physically copied by the threads to assure they
%    are placed in thread-local memory banks
%  - if blocksize>1, matrices are converted to Blocked Compressed Row
%    Storage [1]. This decreases the memory requirements and speeds up SpMV
%    significantly.
%  - if remove_zero_cols==1, empty columns in local matrix parts are
%    removed. This is useful when dealing with large numbers of threads and
%    matrices partitioned using e.g., Metis. [1]
%  - if interleave==1, the matrix is converted to Interleaved Compressed
%    Row Storage [1]. This improves performance of SpMV on some
%    architectures (e.g. Amd).
%  - IMPORTANTLY, if symmetric==1 the matrix A is assumed to be symmetric
%    with only the lower-triangular part given in spA. This almost halves the
%    storage requirements of the matrix and speeds up SpMV up to two times.
%
%Examples:
%
%   % You need ELEMS array, which describes the elements
%   % and Aelem_s, which holds the symmetric parts of element matrices
%
%   % create a symbolic sparse matrix and compute RCM reordering
%   Aconn = sparse_create(ELEMS);
%   perm  = double(mrcm(Aconn));
%   
%   % create a symmetric sparse matrix and permute it
%   opts.symmetric = 1;
%   As = sparse_create(ELEMS, Aelem_s, opts);
%   As = cs_symperm(As',perm)';  % note the SuiteSparse cs_symperm
%
%   % convert the symmetric sparse matrix, use 4 threads
%   opts.nthreads = 4;
%   Ac = sparse_convert(As, opts);
%
%More examples: 
%  edit ex_spmv
%
%See also: SPMV, SPARSE_CREATE
%
%References:
%[1] Krotkiewski, M. and M. Dabrowski. Parallel symmetric sparse matrix-vector product 
%on scalar multi-core cpus. Parallel Computing, 36(4):181–198,  2010.
%

% Copyright 2012, Marcin Krotkiewski, University of Oslo

error ('MEX function not found');

end
