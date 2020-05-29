function [elements, nodes] = CreateMesh2(nelx,nely,nelz,dTE,dTN)
% CREATEMESH2 is a simple mesher of a unit cubic with configurable and structured
% discretization with Hex8 elements. It is faster than CREATEMESH.
% 
%  INPUT:
%   nelx, nely, nelz:   Number of elements on X-Y-Z direction
%   dTN:                Data precision for "nodes" ['single' or 'double']
%   dTE:                Data precision for "elements" ['uint32','uint64']
%
%  OUTPUT:
%   elements:           Conectivty matrix of the elements [nelx8]
%   nodes:              X,Y,Z coordinates of the nodes [nnodx3]
%
%   See also CREATEMESH
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  08/02/2020. Version: 1.0

%% INPUTS CHECK
if ~( mod(nelx,1)==0 && mod(nely,1)==0 && mod(nelz,1)==0 )   % Check if inputs "nel" are integers
    error('Error. Inputs "nelx", "nely" and "nely" must be integers');
elseif ( nelx<0 || nely<0 || nelz<0 )                        % Check if "nel" are positives
    error('Error. Inputs "nelx", "nely" and "nely" must be positives');
elseif ~( strcmp(dTN,'single') || strcmp(dTN,'double') )
    error('Error. Input "dTN" must be "single" or "double"');
elseif ~( strcmp(dTE,'uint32') || strcmp(dTE,'uint64') )
    error('Error. Input "dTE" must be "uint32", "uint64"');
end

%% NODES
sX = ones(1,1,dTN); sY = sX; sZ = sX; % Size of the paralelepiped on X-Y-Z direction
selX = sX/nelx; selY = sY/nely; selZ = sZ/nelz; % Element size
[X,Y,Z]= ndgrid(selX*(0:nelx),selY*(0:nely),selZ*(0:nelz));
nodes = [X(:), Y(:), Z(:)];

%% ELEMENTS
if strcmp(dTE,'uint32')
    nx = uint32(nelx+1);
    ny = uint32(nely+1);
    nz = uint32(nelz+1);    
else
    nx = uint64(nelx+1);
    ny = uint64(nely+1);
    nz = uint64(nelz+1);
end
nnod = nx*ny*nz;
n = 1:nnod;                     % Selects all nodes at base of all elements
fx = nx:nx:nnod;                % Nodes at face X=Sx
fy = zeros(1,nz*nx,dTE);        % Nodes at face Y=Sy
nfy = nx*ny-ny:nx*ny:nnod;      % Node at edge X=0, Y=Sy
for i=1:nz
    fy((i-1)*nx+1:(i-1)*nx+nx) = (1:nx)+(nfy(i)-1);
end
fz = nnod-nx*ny+1:nnod;         % Nodes at face Z=1
n1 = setdiff(n',[fx,fy,fz]');   % Node 1 of each element
elements = [n1, n1+1, n1+nx+1, n1+nx, n1+(nx*ny), n1+(nx*ny)+1, n1+(nx*ny)+nx+1, n1+(nx*ny)+nx];

fprintf('\tMesh created successfully with %u Hex8 elements and %u nodes!\n\n',nelx*nely*nelz,nnod);
