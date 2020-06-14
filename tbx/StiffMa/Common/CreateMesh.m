function [elements, nodes] = CreateMesh(nelx,nely,nelz,dTypeE,dTypeN)
% CREATEMESH is a simple mesher of a unit cubic with configurable and structured
% discretization with Hex8 elements
%  INPUT:
%   nelx, nely, nelz:   Number of elements on X-Y-Z direction
%   dTypeN:             Data precision for "nodes" ['single' or 'double']
%   dTypeE:             Data precision for "elements" ['uint32','uint64', etc]
%
%  OUTPUT:
%   elements:           Conectivty matrix of the elements [nelx8]
%   nodes:              X,Y,Z coordinates of the nodes [Nx3]
%
%   See also NDGRID
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 14/12/2019. Version: 1.4. No plots
% 	Modified: 21/01/2019. Version: 1.3
%   Created:  30/11/2018. Version: 1.0

%% INPUTS CHECK
if ~( mod(nelx,1)==0 && mod(nely,1)==0 && mod(nelz,1)==0 )   % Check if inputs "nel" are integers
    error('Error. Inputs "nelx", "nely" and "nely" must be integers');
elseif ( nelx<0 || nely<0 || nelz<0 )                        % Check if "nel" are positives
    error('Error. Inputs "nelx", "nely" and "nely" must be positives');
elseif ~( strcmp(dTypeN,'single') || strcmp(dTypeN,'double') )
    error('Error. Input "dTypeN" must be "single" or "double"');
elseif ~( strcmp(dTypeE,'int32') || strcmp(dTypeE,'uint32') || strcmp(dTypeE,'int64') || ...
        strcmp(dTypeE,'uint64') || strcmp(dTypeE,'double') )
    error('Error. Input "dTypeE" must be "int32", "uint32", "int64", "uint64" or "double"');
end

%% NODES
sX = ones(1,1,dTypeN); sY = sX; sZ = sX; % Size of the paralelepiped on X-Y-Z direction
selX = sX/nelx; selY = sY/nely; selZ = sZ/nelz; % Element size
[X,Y,Z]= ndgrid(selX*(0:nelx),selY*(0:nely),selZ*(0:nelz));
nodes = [X(:), Y(:), Z(:)];

%% ELEMENTS
elements = zeros(nelx*nely*nelz, 8, dTypeE);
nx = nelx+1; ny = nely+1; el = 0;
for elz= 1:nelz
    for ely = 1:nely
        for elx = 1:nelx
            n1 = nx*ny*(elz-1)+nx*(ely-1)+elx;
            n5 = nx*ny*elz+nx*(ely-1)+elx;
            el = el + 1;
            elements(el,:)=[n1, n1+1, n1+nx+1, n1+nx, n5, n5+1, n5+nx+1, n5+nx];
        end
    end
end
fprintf('\nThe mesh was created successfully with %u Hex8 elements and %u nodes!\n',el,size(nodes,1));
