function [T, WS, stats] = tsearch2(NODES, TRI, POINTS, WS, T, opts)
%TSEARCH2 locates points in triangles and tetrahedrons
%This function works for arbitrary triangulations, not necessarily
%Delaunay triangulations. The syntax is different from that of MATLABs 
%tsearch
%
%  [T, WS, stats] = TSEARCH2(NODES, TRI, POINTS, [WS], [T], [opts])
%
%TSEARCH2 uses quad-tree structure to quickly locate the triangle
%containing a given point. First, qtree structure is built for the centers
%of triangular elements. Next, quadtree('locate', ...) is invoked 
%to quickly find an approximate point location, i.e. some nearby triangle. 
%The enclosing triangle is then found by traversing the list of triangle 
%neighbors. 
%
%The algorithm is the same in 3D, except an octree structure is used.
%
%Arguments:
%  NODES          : coordinates of points in the triangulation [ndim X nnodes]
%  TRI            : triangulation of NODES [ndim+1 X ntriangles]
%  POINTS         : coordinates of points to be located [ndim X npoints]
%  WS             : Optional workspace structure containing the following fields:
%   WS.NEIGHBORS  : a [ndim+1 X nel] array of triangle/tet neigbors. Order of
%                   neighbors matters: neighbor 1 is opposite node 1, etc.
%                   If there is no corresponding neighbor, NEIGHBORS should 
%                   contain 0. See HELP QUADTREE for details. 
%                   Optional.
%                   If not given, computed - incurs overhead.
%
%   WS.ELEMS_CENTERS: coordinates of the central points of the triangles. 
%                   Optional.
%                   If not given, computed - incurs overhead.
%
%   WS.xmin       : Spatial domain extents of the MESH. Optional. 
%   WS.xmax         If not given, computed - incurs overhead.
%   WS.ymin
%   WS.ymax          
%   WS.zmin
%   WS.zmax          
%                     
%
%   WS.qtree      : Quad-(Oct-)tree structure created by an earlier call to
%                   TSEARCH2. qtree can be reused between searches done on 
%                   the same mesh, which may speed things up depending on the
%                   problem size. Optional.
%                   If not given, computed - incurs overhead.
%  T              : Optional. Initial guess regarding the location of the
%                   points in TRI. Can be reused for subsequent calls to
%                   tsearch2 if the locations of the points do not change
%                   too much.  
%
%  opts           : optional structure containing the following fields:
%    opts.nthreads[=0]      number of worker threads for the parallel
%                           part of the code. 0 means the value is taken
%                           from the environment variable OMP_NUM_THREADS
%    opts.verbosity[=0]     display some internal timing information
%    opts.cpu_affinity[=0]  use thread to CPU binding for parallel execution
%    opts.cpu_start[=0]     bind first thread to CPU cpu_start
%    opts.inplace[=0]       If inplace is a non-zero integer, the input T 
%                           is modified in-place, i.e. the output array is 
%                           the exact same memory area as the input array.
%                           WARNING: non-conforming MATLAB. Do not use this
%                           unless you are certain that T is never assigned
%                           to any other variable.
%Output:
%
%  T                Triangle IDs. 0 if no triangle contains a given point.
%  WS               Updated workspace, as in input. All required fields are
%                   computed
%  stats            Useful point location statistics, e.g. number of
%                   point-in-triangle tests.
%
%Note that the input/output types are different than in tsearch: 
%T and TRI are uint32.
%
%Examples:
%  edit ex_tsearch2
%
%See also: QUADTREE, TSEARCH, PointLocation

% Copyright 2012, Marcin Krotkiewski, University of Oslo

%% Check number of parameters, their types and sizes
% Minimum and maximum number of parameters
error(nargchk(3, 6, nargin, 'struct'))

% Optional parameters - check if supplied, set to [] if not.
if nargin < 4;  WS  = []; end
if nargin < 5;  T  = uint32([]); end
if nargin < 6;  opts  = struct(); end

% find out the dimension
ndim = size(NODES, 1);
if ndim~=2 && ndim~=3
    error('tsearch2 is supported in 2 and 3 dimensions');
end

% Check types of all parameters. Syntax similar to validateattributes
if ~isempty(ver('matlab'))
    validateattributes(NODES,  {'double'}, {'size', [ndim NaN]});
    validateattributes(TRI,    {'uint32'},  {'size', [ndim+1 NaN]});
    validateattributes(POINTS, {'double'}, {'size', [ndim NaN]});
end

if ~isfield(WS, 'NEIGHBORS')
    WS.NEIGHBORS = find_elem_neighbors(TRI, NODES);
end

if ~isfield(WS, 'xmin') || isempty(WS.xmin)
    WS.xmin = min(NODES(1,:));
end
if ~isfield(WS, 'xmax') || isempty(WS.xmax)
    WS.xmax = max(NODES(1,:));
end

if ~isfield(WS, 'ymin') || isempty(WS.ymin)
    WS.ymin = min(NODES(2,:));
end
if ~isfield(WS, 'ymax') || isempty(WS.ymax)
    WS.ymax = max(NODES(2,:));
end

if ndim==3
    if ~isfield(WS, 'zmin') || isempty(WS.zmin)
        WS.zmin = min(NODES(3,:));
    end
    if ~isfield(WS, 'zmax') || isempty(WS.zmax)
        WS.zmax = max(NODES(3,:));
    end
end

if ~isfield(WS, 'qtree'); WS.qtree = []; end

if ~isempty(T)
    validateattributes(T, {'uint32'}, {'vector'});
end


%% Work
% the below can be removed and is only here for tsearch compatibility
MESH.NODES = NODES;
MESH.ELEMS = TRI;
MESH.NEIGHBORS = WS.NEIGHBORS;

% Is a quadtree structure supplied?
nel = size(TRI, 2);
if ~isempty(WS.qtree);
    % verify that qtree was created for a system 
    % with the same number points
    if nel~=WS.qtree.n_points
        error('qtree data structure is inconsistent with passed triangulation.');
    end
else
    if ~isfield(WS, 'ELEMS_CENTERS')
        if ndim==2
            WS.ELEMS_CENTERS = [...
                mean(reshape(MESH.NODES(1, MESH.ELEMS), 3, nel));...
                mean(reshape(MESH.NODES(2, MESH.ELEMS), 3, nel))];
        else
            WS.ELEMS_CENTERS = [...
                mean(reshape(MESH.NODES(1, MESH.ELEMS), 4, nel));...
                mean(reshape(MESH.NODES(2, MESH.ELEMS), 4, nel));...
                mean(reshape(MESH.NODES(3, MESH.ELEMS), 4, nel))];
        end
    end
    if ndim==2
        WS.qtree = quadtree('create', WS.ELEMS_CENTERS, WS.xmin, WS.xmax, WS.ymin, WS.ymax, 2);
    else
        WS.qtree = octree('create', WS.ELEMS_CENTERS, WS.xmin, WS.xmax, WS.ymin, WS.ymax, WS.zmin, WS.zmax, 2);
    end
end

% location with the help of quadtree and element neigobors
if ndim==2
    [T, stats] = quadtree('locate', WS.qtree, MESH, POINTS, T, opts);
else
    [T, stats] = octree('locate', WS.qtree, MESH, POINTS, T, opts);
end

end

function NEIGHBORS = find_elem_neighbors(TRI, NODES)

if exist('triangulation')
    
    % use MATLAB's CGAL interface
    trep = triangulation(double(TRI'), NODES');
    NEIGHBORS = trep.neighbors()';
    NEIGHBORS(isnan(NEIGHBORS)) = 0;
    NEIGHBORS = uint32(NEIGHBORS);
else
    
    if size(NODES, 1)==3
        error('Can not compute elements NEIGHBORS in 3D. Please provide that information');
    end
    
    % compute by hand
    edges = [2 3 1 3 1 2];
    nel = size(TRI, 2);
    nedges = 3;
    
    % for every element, map element edges in correct order
    % onto the element id * number of edges in element
    elems = repmat(1:nel*nedges, 2, 1);
    EDGES_ELEMS = sparse(double(TRI(edges, :)), elems, 1);
    
    % find elements that use a given edge. ELEMS_ELEMS contains 2
    % for elements, which contain both edge nodes. That is, the element checked
    % and the neighboring element.
    ELEMS_ELEMS = EDGES_ELEMS'*EDGES_ELEMS;
    
    % Remove diagonal entries - always 2 on the diagonal
    ELEMS_ELEMS = ELEMS_ELEMS - 2*speye(nel*nedges, nel*nedges);
    
    [I, J, V] = find(ELEMS_ELEMS);
    %V=2 element's neighbors through segments
    %V=1 element's neighbors through nodes
    
    indx = find(V==2);
    I = I(indx);
    J = J(indx);
    
    NN = accumarray(J,I,[nel*3, 1]);
    NN = reshape(ceil(NN/3), 3, nel);
    NEIGHBORS = uint32(NN);
end
end
