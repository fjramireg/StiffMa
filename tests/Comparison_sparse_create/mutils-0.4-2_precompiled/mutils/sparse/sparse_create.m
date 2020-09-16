function [A, map] = sparse_create(varargin)
%SPARSE_CREATE creates a Finite Element sparse matrix for a given mesh
%
%  A = SPARSE_CREATE(ELEMS, [Aelems=1], [opts], [dof_map=[]])
%
%Arguments:
%  ELEMS          : definition of elements (nnodel x nel)
%  Aelems         : element matrices.
%                   1 - builds a symbolic nnz structure (1 in every non-zero
%                   entry)
%                   a single element matrix - use this if element matrices
%                   are the same for all elements
%                   one element matrix per element - use this if element
%                   matrices are different for all elements
%                   See examples below.
%  dof_map          supply a node/dof map, e.g. node permutation, or
%                   periodicity map
%
%  opts           : structure containing the following fields:
%    opts.n_row_entries[=-1]   average number of entries per row in 
%                              the sparse matrix. Pass -1 to use the default 
%                              value for recognized elements.
%    opts.n_node_dof[=1]       number of degrees of freedom per node. 
%    opts.symmetric[=0]        create symmetric (1) or general (0) sparse matrix.
%    opts.gen_map[=0]          generate a map that maps entries in Aelems to
%                              matrix non-zero entries.
%    opts.nthreads[=0]         number of worker threads for the parallel
%                              part of the code. 0 means the value is taken
%                              from the environment variable OMP_NUM_THREADS
%    opts.verbosity[=0]        display some internal timing information
%    opts.cpu_affinity[=0]     use thread to CPU binding for parallel execution
%    opts.cpu_start[=0]        bind first thread to CPU cpu_start
%
%Output:
%  A               : sparse matrix
%
%Examples:
%
%  connectivity graph
%
%      A = SPARSE_CREATE(ELEMS);
%
%  symmetric sparse, different element matrix for all elements:
%
%      opts.symmetric = 1;
%      Aelem = ones(size(ELEMS,1)*(size(ELEMS,1)+1)/2, size(ELEMS,2));
%      A = SPARSE_CREATE(ELEMS, Aelem, opts);
%
%  symmetric sparse, common element matrix for all elements:
%
%      opts.symmetric = 1;
%      Aelem = ones(size(ELEMS,1)*(size(ELEMS,1)+1)/2, 1);
%      A = SPARSE_CREATE(ELEMS, Aelem, opts);
%
%  general sparse different element matrix for all elements:
%
%      Aelem = ones(size(ELEMS,1)^2, size(ELEMS,2));
%      A = SPARSE_CREATE(ELEMS, Aelem);
%
%  general sparse, common element matrix for all elements:
%
%      Aelem = ones(size(ELEMS,1)^2, 1);
%      A = SPARSE_CREATE(ELEMS, Aelem);

% Copyright 2012, Marcin Krotkiewski, University of Oslo

error ('MEX function not found');

end
