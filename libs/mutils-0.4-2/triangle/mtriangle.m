function [MESH, VORO] = mtriangle(opts, tristr)
%MTRIANGLE creates an unstructred adaptive mesh using triangle by Jonathan Shewchuk
%
%  [MESH, VORO] = MTRIANGLE(opts, triangle_input)
%
%Arguments:
%  opts is a structure containing options that influence triangle. Below
%  are the options and their default values
%
%   opts.element_type     = 'tri3' % use 'tri3', 'tri6', or 'tri7'
%   opts.triangulate_poly = 1;     % Triangulates a Planar Straight Line Graph
%   opts.gen_edges        = 0;     % 1: return edge information
%   opts.gen_neighbors    = 0;     % 1: return element neighbor information
%   opts.gen_elmarkers    = 1;     % 1: return element markers
%   opts.gen_boundaries   = 1;     % 1: return node markers
%   opts.min_angle        = 15;    % minimum triangle angle
%   opts.max_tri_area     = 0;     % maximum triangle area
%   opts.ignore_holes     = 0;     %
%   opts.exact_arithmetic = 1;     % 0: do not use Triangle's exact arithmetic
%   opts.zero_numbering   = 0;     % 1: use zero-based index numbering
%   opts.other_options    = '';    % other triangle options
%
%  triangle_input is a structure defining the input data passed to
%  triangle. For a complete explanation read the triangle documentation.
%
%   tristr.points
%   tristr.segments
%   tristr.segmentmarkers
%   tristr.regions
%   tristr.pointmarkers
%   tristr.pointattributes
%   tristr.holes
%   tristr.triangles
%   tristr.triangleattributes
%   tristr.trianglearea
%
%Output:
%  MESH         mesh structure with the following fields
%   NODES
%   ELEMS
%   elem_markers
%   node_markers
%   EDGES
%   edge_markers
%   SEGMENTS
%   segment_markers
%   NEIGHBORS
%
%  VORO         voronoi diagram structure with the following fields
%   NODES
%   EDGES
%   NORMALS
%
%Note: for Voronoi diagrams, infinite rays are marked as 0 in the EDGES array, 
%not as -1, as originally done by triangle. 
%
%Example:
%
%  points   = [0 0; 1 0; 1 1; 0 1]';
%  segments = [1 2; 2 3; 3 4; 4 1]';
%  opts.element_type     = 'tri3';
%  opts.min_angle        = 30;
%  opts.max_tri_area     = 0.001;
%  tristr.points         = points;
%  tristr.segments       = uint32(segments);
%  MESH = mtriangle(opts, tristr);
%
%More examples:
%  edit ex_triangle

% Copyright 2012, Marcin Krotkiewski, University of Oslo

%% check number of parameters, their types and sizes
% number of parameters
error(nargchk(2, 2, nargin, 'struct'))

if triangle_standalone()   
    [opts, tristr] = set_default_args(opts, tristr);
else
    opts = mvalidateattributes(opts, {'struct'}, {}, ...
        {'model_name', 'model'}, {'char'}, {'vector'}, ...
        {'element_type', 'tri3'}, {'char'}, {'vector'}, ...
        {'triangulate_poly', 1}, {'numeric'}, {'scalar'}, ...
        {'gen_edges', 0}, {'numeric'}, {'scalar'}, ...
        {'gen_neighbors', 0}, {'numeric'}, {'scalar'}, ...
        {'gen_elmarkers', 1}, {'numeric'}, {'scalar'}, ...
        {'gen_boundaries', 1}, {'numeric'}, {'scalar'}, ...
        {'min_angle', 15}, {'numeric'}, {'scalar'}, ...
        {'max_tri_area', 0}, {'numeric'}, {'scalar'}, ...
        {'ignore_holes', 0}, {'numeric'}, {'scalar'}, ...
        {'exact_arithmetic', 1}, {'numeric'}, {'scalar'}, ...
        {'zero_numbering', 0}, {'numeric'}, {'scalar'}, ...
        {'other_options', ''}, {'char'}, {'vector'});
    
    tristr = mvalidateattributes(tristr, {'struct'}, {}, ...
        {'points', []}, {'double'}, {'size' [2 NaN] 'or empty'}, ...
        {'segments', []}, {'uint32'}, {'size' [2 NaN] 'or empty'}, ...
        {'regions', []}, {'double'}, {'size' [4 NaN] 'or empty'}, ...
        {'holes', []}, {'double'}, {'size' [2 NaN] 'or empty'}, ...
        {'triangles', []}, {'uint32'}, {'or empty'});
    
    ntrianglenodes = size(tristr.triangles, 1);
    if ~any(ntrianglenodes==[0 3 6])
        merror('tristr.triangles must be of size [3 x ntriangles] or [6 x ntriangles]');
    end
    
    npoints = size(tristr.points, 2);
    nsegments = size(tristr.segments, 2);
    nregions = size(tristr.regions, 2);
    nholes = size(tristr.holes, 2);
    ntriangles = size(tristr.triangles, 2);
    tristr = mvalidateattributes(tristr, {'struct'}, {}, ...
        {'pointmarkers', []}, {'uint32'}, {'nonnegative' 'size' [1 npoints] 'or empty'}, ...
        {'segmentmarkers', []}, {'uint32'}, {'nonnegative' 'size' [1 nsegments] 'or empty'}, ...
        {'pointattributes', []}, {'double'}, {'size' [NaN npoints] 'or empty'}, ...
        {'triangleattributes', []}, {'double'}, {'size' [NaN ntriangles] 'or empty'}, ...
        {'trianglearea', []}, {'double'}, {'size' [1 ntriangles] 'or empty'});

    % region markers must be nonnegative integers
    if nregions>0
        region_markers = tristr.regions(3,:);
        mvalidateattributes(region_markers, {'numeric'}, {'nonnegative' 'integer'});
    end
end

% create triangle options string
tri_flag = generate_options_string(opts);


%% run triangle
if exist(['triangle.' mexext], 'file') == 3
    
    %% use the mex file
    [MESH.NODES, MESH.ELEMS, MESH.elem_markers, MESH.node_markers, ...
        MESH.EDGES, MESH.edge_markers, ...
        MESH.SEGMENTS, MESH.segment_markers, ...
        MESH.NEIGHBORS, ...
        VORO.NODES, VORO.EDGES, VORO.NORMALS] = ...
        triangle(tri_flag, tristr.points, tristr.pointmarkers, tristr.pointattributes, ...
        tristr.segments, tristr.segmentmarkers, tristr.holes, tristr.regions, ...
        tristr.triangles, tristr.triangleattributes, tristr.trianglearea);
else
    
    %% triangle executable only supported in milamin, 
    %  not in standalone distribution
    if triangle_standalone()
        error('MEX function not found');
    end
    
    %  first find triangle in the path
    if exist('triangle', 'file') == 2
        texec = 'triangle';
        print_message('Using triangle found in the path.\n');
    else
        global milamin_data;
        if ispc
            texec = [milamin_data.path 'ext/triangle/triangle.exe'];
        else
            texec = [milamin_data.path 'ext/triangle/triangle.' lower(computer)];
        end
        if ~exist(texec)
            error([mfilename ': triangle executable not found: ' texec]);
        end
    end
    
    % delete old files
    delete([opts.model_name '.*.poly']);
    delete([opts.model_name '.*.edge']);
    delete([opts.model_name '.*.ele']);
    delete([opts.model_name '.*.node']);
    delete([opts.model_name '.*.neigh']);
    
    % write input files
    triangle_write(opts.model_name, tristr);
    
    % execute triangle
    [status,result] = system([texec ' -' tri_flag ' ' opts.model_name '.poly']);
    if status
        error([mfilename ': triangle executable failed: ' result]);
    end
    
    % read output files
    [MESH.NODES, MESH.ELEMS, MESH.elem_markers, MESH.node_markers, ...
        MESH.EDGES, MESH.edge_markers, ...
        MESH.SEGMENTS, MESH.segment_markers, ...
        MESH.NEIGHBORS] = ...
        triangle_read([opts.model_name '.1']);
end

if isempty(MESH.SEGMENTS)
    MESH = rmfield(MESH, 'SEGMENTS');
    MESH = rmfield(MESH, 'segment_markers');
end

if isempty(MESH.EDGES)
    MESH = rmfield(MESH, 'EDGES');
    MESH = rmfield(MESH, 'edge_markers');
end
    
if isempty(MESH.elem_markers)
    MESH.elem_markers = zeros(1, length(MESH.ELEMS), 'uint32');
else
    MESH.elem_markers = uint32(MESH.elem_markers);
end

if isfield(MESH, 'EDGES') & isempty(MESH.EDGES)
    MESH = rmfield(MESH, 'EDGES');
end

if isfield(MESH, 'edge_markers') & isempty(MESH.edge_markers)
    MESH = rmfield(MESH, 'edge_markers');
end

if isfield(MESH, 'NEIGHBORS')
    if isempty(MESH.NEIGHBORS)
        MESH = rmfield(MESH, 'NEIGHBORS');
    else
        temp = MESH.NEIGHBORS==intmax('uint32');
        if ~opts.zero_numbering
            MESH.NEIGHBORS(temp)=0;
        end
    end
end

% extra functionality comes with MILAMIN_v2
if ~triangle_standalone()
    MESH = mesh_info(MESH,0);
    
    % - triangle does generate edge information, but does not
    %   create ELEMS_EDGES, i.e., edge-based element definitions
    % - triangle returns only two-node edges/segments.
    %   Hence, segment/edge definitions need to be updated with midedge nodes
    %   for higher-order elements.
    el_info = element_info(opts.element_type);
    if el_info.order>1 | opts.gen_edges
        MESH = mesh_find_edges(MESH);
    end
    if strcmp(opts.element_type, 'tri7')
        MESH = mesh_convert(MESH, opts.element_type);
    end
else
    if strcmp(opts.element_type, 'tri7')
        nel = size(MESH.ELEMS, 2);
        ncorners = 3;
        MESH.ELEMS(end+1,:)  = max(MESH.ELEMS(:))+uint32([1:nel]);
        MESH.NODES = [MESH.NODES [...
            mean(reshape(MESH.NODES(1, MESH.ELEMS(1:ncorners,:)), ncorners, nel));...
            mean(reshape(MESH.NODES(2, MESH.ELEMS(1:ncorners,:)), ncorners, nel))]];
        if isfield(MESH, 'node_markers')
            MESH.node_markers = [MESH.node_markers zeros(1,nel,'uint32')];
        end
    end
end

if ~opts.gen_edges & isfield(MESH, 'EDGES')
    MESH = rmfield(MESH, {'EDGES' 'edge_markers' 'ELEMS_EDGES'});
    if ~triangle_standalone()
        MESH = mesh_info(MESH,0);
    end
end

end


%% parameter analysis functions
function opts_str = generate_options_string(opts)
opts_str = [opts.other_options];
if opts.triangulate_poly
    opts_str = [opts_str 'p'];
end
switch opts.element_type
    case 'tri3'
        opts_str = [opts_str 'o1'];
    case {'tri6' 'tri7'}
        opts_str = [opts_str 'o2'];
    otherwise
        error([mfilename ': unknown triangle element type.']);
end
if opts.gen_edges
    opts_str = [opts_str 'e'];
end
if opts.gen_neighbors
    opts_str = [opts_str 'n'];
end
if opts.gen_elmarkers
    opts_str = [opts_str 'A'];
end
if ~opts.gen_boundaries
    opts_str = [opts_str 'B'];
end
opts_str = [opts_str 'q' num2str(opts.min_angle, '%.16f')];
if opts.max_tri_area
    opts_str = [opts_str 'a' num2str(opts.max_tri_area, '%.16f')];
end
if opts.ignore_holes
    opts_str = [opts_str 'O'];
end
if ~opts.exact_arithmetic
    opts_str = [opts_str 'X'];
end
if opts.zero_numbering
    opts_str = [opts_str 'z'];
end
end


%% set default options for stand-alone mtriangle
function opts = set_default(opts, field, value)
if ~isfield(opts, field)
    opts.(field) = value;
end
end

function [opts, tristr] = set_default_args(opts, tristr)
    % set default options, no validation
    opts = set_default(opts, 'model_name', 'model');
    opts = set_default(opts, 'element_type', 'tri3');
    opts = set_default(opts, 'triangulate_poly', 1);
    opts = set_default(opts, 'gen_edges', 0);
    opts = set_default(opts, 'gen_neighbors', 0);
    opts = set_default(opts, 'gen_elmarkers', 1);
    opts = set_default(opts, 'gen_boundaries', 1);
    opts = set_default(opts, 'min_angle', 15);
    opts = set_default(opts, 'max_tri_area', 0);
    opts = set_default(opts, 'ignore_holes', 0);
    opts = set_default(opts, 'exact_arithmetic', 1);
    opts = set_default(opts, 'zero_numbering', 0);
    opts = set_default(opts, 'other_options', '');
    
    % set default empty tristr fields
    tristr = set_default(tristr, 'points', []);
    tristr = set_default(tristr, 'segments', []);
    tristr = set_default(tristr, 'regions', []);
    tristr = set_default(tristr, 'holes', []);
    tristr = set_default(tristr, 'triangles', []);
    tristr = set_default(tristr, 'pointmarkers', []);
    tristr = set_default(tristr, 'segmentmarkers', []);
    tristr = set_default(tristr, 'pointattributes', []);
    tristr = set_default(tristr, 'triangleattributes', []);
    tristr = set_default(tristr, 'trianglearea', []);
end


%% do we run from inside milamin, or stand-alone?
function retval = triangle_standalone
% milamin_data is a global variable defined in milamin_init
global milamin_data;
if ~isempty(milamin_data)
    retval = 0;
else
    retval = 1;
end
end
