function qtree = quadtree(varargin)
%QUADTREE is a 2D quad-tree implementation
%
%  qtree = QUADTREE('create', POINTS, [xmin=0], [xmax=1], [ymin=0], [ymax=1], [max_points_in_quadrant=1])
%  [T, stats] = QUADTREE('locate', qtree, MESH, MARKERS)
%  [T, stats] = QUADTREE('locate', qtree, MESH, MARKERS, T)
%  [T, stats] = QUADTREE('locate', qtree, MESH, MARKERS, T, opts)
%  I = QUADTREE('reorder', qtree)
%  QUADTREE('vtkwrite', qtree, [file_name])
%
%This function provides quad-tree creation, point-in-triangle location for 
%unstructured 2D triangular meshes, point ordering extraction, and VTK 
%output of the quad-tree.
%
%Use scenarios
% 1. Quad-tree creation
%
%  qtree = QUADTREE('create', POINTS, [xmin=0], [xmax=1], [ymin=0], [ymax=1], [max_points_in_quadrant=1])
%
%  Creates the quad-tree structure based on a cloud of points
%  
%  Arguments:
%    POINTS         points (2 x npoints, double).
%    xmin,xmax
%    ymin,ymax      Domain extents, by default from 0 to 1. 
%    max_points_in_quadrant  Controls the quad-tree refinement level. 
%                   The quadrants are split only after the number of points 
%                   in a quadrant exceeds this value.
%
%  Output:
%    qtree          quad-tree strucutre
%
%  The quad-tree is not restricted by the 2:1 rule, i.e., the difference in
%  refinement level between neighboring quadrants can be greater than 1.
%
% 2. Quad-tree based point-in-triangle location
%
%  [T, stats] = QUADTREE('locate', qtree, MESH, MARKERS)
%
%  For quad-trees created based on element centers of an unstrucutred
%  triangular MESH, this call finds the triangles containing MARKERS. 
%  Triangle location in done parallel on SMP computers using OpenMP. 
%  Set the OMP_NUM_THREADS variable to the desired number of CPU cores.
%
%  Algorithm:
%  For every marker the quadrant containing that marker is found in the 
%  quad-tree structure. All mesh elements 'contained' in that quadrant are 
%  considered to be 'close' to the marker being located. Searching for the 
%  exact containing triangle is then done using the Green and Sibson algorithm.
%  For that, neighbor information is required for every triangle. Termination 
%  of the algorithm is assured by verifying that every triangle is searched at
%  most once. In the worst-case scenario, e.g. locating MARKERS that do not 
%  lie in any triangle, all elements are tested for every such marker. 
%  In a usual scenario, for a given value of max_points_in_quadrant only 
%  a small and constant number of triangles is tested for every marker.
%
%  Arguments:
%   qtree      quad-tree returned by quadtree('create', ...);
%   MESH       a structure that must have the following fields:
%
%     NODES      mesh node coordinates
%                size: 2 x nnodes
%                type: double
%     ELEMS      standard FEM triangular element definitions, 
%                first 3 nodes are corner nodes in anti-clockwise order
%                size: nnodel x nel
%                type: uint32
%     NEIGHBORS  contains element's neighbors for every element. Order of
%                neighbors matters: neighbor 1 is opposite node 1, etc.
%                If there is no corresponding neighbor, NEIGHBORS should contain 0.
%                size: 3 x nel
%                type: uint32
%   MARKERS      marker coordinates
%                size: 2 x nmarkers
%
%  Output:
%   T          elements containing MARKERS. 0 if no triangle contains a given point.
%   stats      useful search statistics, e.g. number of triangles searched.
%
% 3. Quad-tree based point-in-triangle location with approximate locations
%
%  [T, stats] = QUADTREE('locate', qtree, MESH, MARKERS, T)
%
%  Approximate/incomplete MARKER locations can be supplied as input. In this case
%  QUADTREE performs only the Green and Sibson search starting from the element 
%  given in T. For markers, for which T contains 0, QUADTREE performs 
%  a standard quad-tree search followed by the mesh search.
%
% 4. Quad-tree based point-in-triangle location with additional options
%
%  [T, stats] = QUADTREE('locate', qtree, MESH, MARKERS, T, opts)
%
%  opts           : structure containing the following fields:
%    opts.inplace[=0]       If inplace is a non-zero integer, the input T 
%                           is modified in-place, i.e. the output array is 
%                           the exact same memory area as the input array. 
%    opts.nthreads[=0]      number of worker threads for the parallel
%                           part of the code. 0 means the value is taken
%                           from the environment variable OMP_NUM_THREADS
%    opts.verbosity[=0]     display some internal timing information
%    opts.cpu_affinity[=0]  use thread to CPU binding for parallel execution
%    opts.cpu_start[=0]     bind first thread to CPU cpu_start
%
%  WARNING! The inplace feature should be used with extreme caution since such 
%  behavior is not encouraged by MATLAB. Other variables that 'link' to T 
%  would be seamlessly modified leading to wrong results and unexpected errors.
%  Using this syntax make sure that T has not been assigned to other
%  variables, and that no variables were assigned to T.
%
% 5. Quad-tree based node reordering
%
%  I = QUADTREE('reorder', qtree)
%
%  Returns ordering of points as they appear in the quad-tree.
%
%
% 6. VTK output
%
%  QUADTREE('vtkwrite', qtree, [file_name])
%
%  Writes a VTK file to visualize the quad-tree.
%
%Examples:
%  edit ex_quadtree
%
%See also: TSEARCH2

% Copyright 2012, Marcin Krotkiewski, University of Oslo

error ('MEX function not found');

end
