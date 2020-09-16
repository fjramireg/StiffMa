function r = spmv(varargin)
%SPMV performs a parallel Sparse Matrix - Vector Multiplication (SpMV)
%
%  r = SPMV(spA, x, [rowdist], [opts]);
%
%SPMV multiplies the transpose of spA by the vector x:
%
%  r = spA'*x;
%
%There are two ways to use SPMV:
%  1. multiply native MATLAB sparse matrix by a vector in parallel
%
%  Arguments:
%    spA            : square sparse matrix
%    x              : vector
%    [rowdist]      : optional, row distribution of matrix among the threads. 
%                     rowdist is a vector of size [1 x nthreads+1]. Entry i holds
%                     the starting row number for thread i. The last entry in
%                     rowdist is the dimension of the matrix + 1.
%  opts               structure containing optional arguments
%
%    opts.nthreads[=0]      number of worker threads for the parallel
%                           part of the code. 0 means the value is taken
%                           from the environment variable OMP_NUM_THREADS
%    opts.verbosity[=0]     display some internal timing information
%    opts.cpu_affinity[=0]  use thread to CPU binding for parallel execution
%    opts.cpu_start[=0]     bind first thread to CPU cpu_start
%
%
%  2. multiply a matrix converted by SPARSE_CONVERT by a vector in parallel
%
%  Arguments:
%    spA            : square sparse matrix obtained by SPARSE_CONVERT
%    x              : vector
%
%  Examples:
%
%    % parallel spmv for converted matrices
%    opts.nthreads = 4;
%    Ac = sparse_convert(A, opts);
%    v0 = spmv(Ac, x);
%
%    % parallel spmv for native matlab matrices
%    setenv('OMP_NUM_THREADS', opts.nthreads);
%    v1 = spmv(A, x);
%
%    % native sequential matlab
%    v2 = A*x;
%    norm(v1-v2)
%    norm(v0-v2)
%
%More examples: 
%    edit ex_spmv
%
%See also: SPARSE_CONVERT, SPARSE_CREATE
%
%References:
%[1] Krotkiewski, M. and M. Dabrowski. Parallel symmetric sparse matrix-vector product 
%on scalar multi-core cpus. Parallel Computing, 36(4):181–198,  2010.
 
% Copyright 2012, Marcin Krotkiewski, University of Oslo

error ('MEX function not found');

end
