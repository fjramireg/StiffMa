%% Using quad trees to locate points in triangles
% ex_tsearch2 shows how to locate randomly placed markers in triangular
% elements of an unstructured mesh. Rough performance comparison between
% tsearch2 and MATLABs tsearch and PointLocation is included.
%%

%% Generate Delaunay triangulation of random points

% setup domain - rectangular box
x_min        = -1; 
x_max        =  1;
y_min        = -2; 
y_max        =  1;
z_min        = -1;
z_max        = -2;

points   = [
    x_min y_min z_min;
    x_max y_min z_min;
    x_max y_max z_min;
    x_min y_max z_min;
    x_min y_min z_max;
    x_max y_min z_max;
    x_max y_max z_max;
    x_min y_max z_max;]';

% randomize points inside the domain
n_points = 5e4;
rpoints  = rand(3, n_points);
rpoints(1,:) = rpoints(1,:)*(x_max-x_min) + x_min;
rpoints(2,:) = rpoints(2,:)*(y_max-y_min) + y_min;
rpoints(3,:) = rpoints(3,:)*(z_max-z_min) + z_min;
points   = [points rpoints];

%% 
% First, create a new Delaunay triangulation. |*tsearch2*| does not require 
% a Delaunay triangulation, but MATLABs |*tsearch*| and |*pointLocation*| do.
% This part is done much faster with triangle. However, MATLAB
% |pointLocation| routine used later in comparison requires 
% a Delaunay triangulation created by |DelaunayTri| only.

DT   = DelaunayTri(points');
trep = triangulation(DT.Triangulation, points');


%% Generate randomly placed markers
n_markers = 1e6;
markers   = rand(3, n_markers);
markers(1,:) = markers(1,:)*(x_max-x_min) + x_min;
markers(2,:) = markers(2,:)*(y_max-y_min) + y_min;
markers(3,:) = markers(3,:)*(z_max-z_min) + z_min;


%% Point location using |tsearch2|
% |*tsearch2*| uses quad-tree structure to quickly locate the triangle
% containing a given point. First, qtree structure is built for the centers
% of triangular elements. Next, |*quadtree('locate', ...)*| is invoked 
% to find an approximate point location, i.e. some nearby triangle. 
% The enclosing triangle is then found by traversing the list of triangle 
% neighbors.

% Set important tsearch2 parameters
WS = [];
WS.NEIGHBORS = trep.neighbors()';        % we need triangle neighbors
WS.NEIGHBORS(isnan(WS.NEIGHBORS)) = 0;
WS.NEIGHBORS = uint32(WS.NEIGHBORS);
WS.xmin = x_min;  % the domain extents need to be specified
WS.xmax = x_max;
WS.ymin = y_min;
WS.ymax = y_max;
WS.zmin = z_min;
WS.zmax = z_max;

% tsearch2 requires the triangulation to be of type uint32 
ELEMS = uint32(DT.Triangulation');

t = tic;
[T1, WS, stats] = tsearch2(points, ELEMS, markers, WS);
display(['tsearch2 (sequential): ' num2str(toc(t))]);

%%
% |stats| provide useful statistics about the point location

stats

%%
% * |avg_elems_searched| says how many point-in-triangle tests have been
% performed on average for every marker.
%
% * |n_max_elems_searched| gives the maximum number of point-in-triangle tests
% performed for one marker
%
% The quadtree approach is very effective - on average around 2 triangles
% need to be checked in order to find the enclosing triangle.
%
% If the marker locations do not change too much in time, 
% |T1| can be used as an initial guess for |tsearch2|. In that case quadtree 
% search is not performed. Instead, the enclosing triangle is found by
% walking through the neighbors of the triangle provided as the initial
% guess.

% run tsearch2 with an exact element map
t=tic;
[T1, ~, stats] = tsearch2(points, ELEMS, markers, WS, T1);
display(['tsearch2 with exact element map (sequential): ' num2str(toc(t))]);

stats

%%
% Here, |avg_elems_searched| is 1 because we provided the exact point location
% vector |T1| as initial guess. This approach can be much faster, because
% the quadtree search is not necessary, and less triangles might need to be
% checked.

%% Parallel point location using |tsearch2|
% The point location is parallelized on shared memory computers using OpenMP. 
% However, creation of the quadtree structure is sequential. 
% Hence, speedups are best when the number of mesh
% elements is much smaller than the number of located markers.
t = tic;
opts.nthreads = 2;
T2 = tsearch2(points, ELEMS, markers, WS, [], opts);
display(['tsearch2 (parallel): ' num2str(toc(t))]);

% compare results, must be identical
if ~isequal(T1, T2)
    display(['WARNING: Maps obtained by sequential and parallel tsearch2 do not match.']);
end


%% Point location using MATLABs |tsearch|
% Note that tsearch is obsolete and has been removed in MATLAB 2012a.
try
    t = tic;
    T3 = tsearch(points(1,:), points(2,:), double(ELEMS'), markers(1,:), markers(2,:), markers(3,:));
    display(['tsearch: ' num2str(toc(t))]);
    
    % compare results - might differ for nodes lying on edges
    ndiff = numel(unique(T1-uint32(T3)))-1;
    if ndiff
        display(['WARNING: Maps obtained by tsearch and tsearch2 do not match. ' ...
            num2str(ndiff) ' different results.']);
    end
catch
    display('WARNING: tsearch does not work or has been removed');
end

%% Point location using MATLABs |pointLocation|
t = tic;
T4 = pointLocation(DT, markers');
display(['pointLocation: ' num2str(toc(t))]);

% compare results - might differ for nodes lying on edges
ndiff = numel(unique(T1-uint32(T4')))-1;
if ndiff
    display(['WARNING: Maps obtained by tsearch2 and pointLocation do not match. ' ...
        num2str(ndiff) ' different results.']);
end
