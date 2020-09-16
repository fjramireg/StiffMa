function [perm,invperm,rowdist] = metis_part(varargin)
%METIS_PART is a MATLAB interface to METIS graph partitioning library
%
%  [perm,invperm,rowdist] = METIS_PART(A, nparts)
%
%Arguments:
%  A             : sparse matrix to be reordered
%  nparts        : number of partitions
%
%Output:
%  perm          : permutation vector
%  invperm       : inverse permutation vector
%  rowdist       : row distribution of the permuted matrix
%
%See also: MRCM, SYMRCM

% Copyright 2012, Marcin Krotkiewski, University of Oslo

error ('MEX function not found');

end
