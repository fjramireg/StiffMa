function V = einterp(varargin)
%EINTERP computes FEM interpolation at given points in an unstructured mesh.
%
%  [V_MARKERS, UV] = EINTERP(MESH, V, MARKERS, ELEMENT_MAP, [opts])
%
%Arguments:
%  MESH           structure containing the mesh description
%   MESH.NODES    mesh nodes coordinates (2 x nnod)
%   MESH.ELEMS    elements definition (nnodel x nel)
%
%  V              nodal values (1 x nnod or 2 x nnod)
%  MARKERS        marker coordinates (2 x n_markers)
%  ELEMENT_MAP    elements containing the markers
%  opts           structure containing optional arguments
%
%    opts.nthreads[=0]      number of worker threads for the parallel
%                           part of the code. 0 means the value is taken
%                           from the environment variable OMP_NUM_THREADS
%    opts.verbosity[=0]     display some internal timing information
%    opts.cpu_affinity[=0]  use thread to CPU binding for parallel execution
%    opts.cpu_start[=0]     bind first thread to CPU cpu_start
%
%Output:
%  V_MARKERES     interpolated values in MARKERS
%  UV             local element coordinates of markers in containing elements
%
%EINTERP uses SSE instructions, prefetching and OpenMP parallelization to
%achieve good performance on modern multi-core CPUs. Set the OMP_NUM_THREADS 
%variable to the desired number of CPU cores.
%
%Currently, the implementation is limited to 7-node triangular elements and
%two degrees of freedom per mesh node. More elements will be added as
%requested by the users.
%
%Elements supported:
%
%EINTERP currently supports interpolation of 1 and 2 degrees of freedom 
%per node for the following elements: tri3, tri7, quad4, quda9.
%
%Examples:
%  edit ex_einterp
%
%See also: TSEARCH, TSEARCH2

% Copyright 2012, Marcin Krotkiewski, University of Oslo

error ('MEX function not found');

end
