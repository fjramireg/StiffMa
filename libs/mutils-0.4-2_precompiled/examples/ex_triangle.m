%% Unstructured mesh generation using triangle MEX function
% ex_triangle script shows how to generate a simple unstructured mesh using
% the triangle MEX function provided by mutils.

%% Generate unstructured triangular mesh
% Setup domain - square box
points   = [0 0; 1 0; 1 1; 0 1]'; % corner points
segments = [1 2; 2 3; 3 4; 4 1]'; % segments

%%
% Set triangle options
opts = [];
opts.element_type     = 'tri7';   % element type
opts.gen_neighbors    = 1;        % generate element neighbors
opts.triangulate_poly = 1;
opts.min_angle        = 30;
opts.max_tri_area     = 0.001;

%%
% Create triangle input structure
tristr.points         = points;
tristr.segments       = uint32(segments);  % note segments have to be uint32

%%
% Generate the mesh using triangle
MESH = mtriangle(opts, tristr);

%%
% Show the mesh
ncorners = 3;
nel = length(MESH.ELEMS);
X = reshape(MESH.NODES(1,MESH.ELEMS(1:ncorners,:)), ncorners, nel);
Y = reshape(MESH.NODES(2,MESH.ELEMS(1:ncorners,:)), ncorners, nel);
figure(1); clf;
h = patch(X, Y, 'g');
axis square
