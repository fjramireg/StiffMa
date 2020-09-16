%% FEM interpolation of velocities in markers
% ex_einterp shows how to 
% 
% * locate a cloud of markers in elements of an unstructured 2D triangular mesh
% * interpolate velocities from element nodes to the markers using FEM
%   interpoolation
%%

function ex_einterp


%% Generate unstructured triangular mesh

% Set triangle options
opts = [];
opts.max_tri_area  = 0.0001;
opts.element_type  = 'tri7';
opts.gen_edges     = 0;
opts.gen_neighbors = 1;

% Setup domain - rectangular box
tristr.points   = [-2 2 2 -2; -1 -1 1 1];
tristr.segments = uint32([1 2 3 4; 2 3 4 1]);

% Generate the mesh using triangle
MESH = mtriangle(opts, tristr);


%% Generate the markers
n_markers = 5e6;

% grid of markers
[X, Y]    = meshgrid(linspace(0,1,ceil(sqrt(n_markers))));
markers   = [X(:) Y(:)]';

% scale coordinates to fit the domain
markers(1,:)   = 4*markers(1,:)-2;
markers(2,:)   = 2*markers(2,:)-1;
n_markers      = length(markers);


%% Locate markers in elements using tsearch2

% Set important tsearch2 parameters
WS = [];
WS.NEIGHBORS = MESH.NEIGHBORS;  % element neighbor information
WS.xmin = -2;                   % domain extents
WS.xmax =  2;
WS.ymin = -1;
WS.ymax =  1;

% Run tsearch2 on 2 CPUs
opts.nthreads = 2;
t=tic;
T = tsearch2(MESH.NODES, MESH.ELEMS(1:3, :), markers, WS, [], opts);
display(['tsearch2: ', num2str(toc(t))]);


%% FEM interpolation of 2D velocities in markers

% Generate random velocity field
V = 1+rand(size(MESH.NODES));

%% 
% Interpolate using einterp MEX function

% Use 1 CPU
opts.nthreads=1;
t = tic;
Vm_seq = einterp(MESH, V, markers, T, opts);
display(['einterp MEX (sequential): ', num2str(toc(t))]);

% Use 2 CPUs
opts.nthreads=2;
t=tic;
Vm = einterp(MESH, V, markers, T, opts);
display(['einterp MEX (parallel): ', num2str(toc(t))]);

% Compare results
if unique(Vm-Vm_seq) ~= 0
    merror('sequential and parallel einterp results differ');
end

% Compute interpolated values and local coordinates of markers (UVm).
% Returning Uvm takes additional time.
t=tic;
[temp, UVm] = einterp(MESH, V, markers, T, opts);
display(['einterp MEX (parallel), returns local coordinates: ', num2str(toc(t))]);
clear temp;


%%
% Interpolate using native MATLAB implementation
t=tic;
eta  = local_coordinates(MESH,markers,T);
eta1 = eta(1,:);
eta2 = eta(2,:);
eta3 = 1-eta1-eta2;

eta1eta2eta3 = eta1.*eta2.*eta3;
N = zeros(n_markers,7);
N(:,1) = eta1.*(2*eta1-1) + 3*eta1eta2eta3;
N(:,2) = eta2.*(2*eta2-1) + 3*eta1eta2eta3;
N(:,3) = eta3.*(2*eta3-1) + 3*eta1eta2eta3;
N(:,4) = 4*eta2.*eta3 - 12*eta1eta2eta3;
N(:,5) = 4*eta1.*eta3 - 12*eta1eta2eta3;
N(:,6) = 4*eta1.*eta2 - 12*eta1eta2eta3;
N(:,7) =                27*eta1eta2eta3;

ELEMS = MESH.ELEMS(:,T);
Vx = V(1,:);
Vy = V(2,:);
vx = sum(Vx(ELEMS).*N');
vy = sum(Vy(ELEMS).*N');
display(['einterp MATLAB: ', num2str(toc(t))]);

% compare results: MEX vs MATLAB
UV = [eta2; eta3];
display(['Maximum difference between MATLAB and MEX implementations (UVm): ' ...
    num2str(norm(UVm(:)-UV(:), 'inf'))]);

Vs = [vx; vy];
display(['Maximum difference between MATLAB and MEX implementations: ' ...
    num2str(norm(Vm(:)-Vs(:),'inf'))]);

end % function ex_einterp


%% Auxiliary functions
% A function that computes local element coordinates of randomly placed
% markers in an unstructured triangular mesh. It is only needed for native
% MATLAB implementation. The einterp MEX function computes the local
% coordinates internally.
function eta = local_coordinates(MESH,points,point_elems)
ndim = size(MESH.NODES, 1);
nnod = size(MESH.NODES, 2);
nel  = length(MESH.ELEMS);

ENOD_X = reshape(MESH.NODES(1,MESH.ELEMS(1:3,:)), 3,nel);
ENOD_Y = reshape(MESH.NODES(2,MESH.ELEMS(1:3,:)), 3,nel);

area  = ENOD_X(2,:).*ENOD_Y(3,:) - ENOD_X(3,:).*ENOD_Y(2,:) + ...
    ENOD_X(3,:).*ENOD_Y(1,:) - ENOD_X(1,:).*ENOD_Y(3,:) + ...
    ENOD_X(1,:).*ENOD_Y(2,:) - ENOD_X(2,:).*ENOD_Y(1,:);

ENOD_X_LONG = ENOD_X(:,point_elems);
ENOD_Y_LONG = ENOD_Y(:,point_elems);

eta = zeros(ndim,length(points));
eta(1,:)  = ENOD_X_LONG(2,:).*ENOD_Y_LONG(3,:) - ENOD_X_LONG(3,:).*ENOD_Y_LONG(2,:) + ...
    ENOD_X_LONG(3,:).*points(2,:) - points(1,:).*ENOD_Y_LONG(3,:) + ...
    points(1,:).*ENOD_Y_LONG(2,:) - ENOD_X_LONG(2,:).*points(2,:);

eta(2,:)  = ENOD_X_LONG(3,:).*ENOD_Y_LONG(1,:) - ENOD_X_LONG(1,:).*ENOD_Y_LONG(3,:) + ...
    ENOD_X_LONG(1,:).*points(2,:) - points(1,:).*ENOD_Y_LONG(1,:) + ...
    points(1,:).*ENOD_Y_LONG(3,:) - ENOD_X_LONG(3,:).*points(2,:);

area_long = area(point_elems);

eta(1,:) = eta(1,:)./area_long;
eta(2,:) = eta(2,:)./area_long;
end
