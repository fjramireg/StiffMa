function [elements, nodes] = CreateMesh(nelx,nely,nelz)
sizeX = 1;
sizeY = 1;
sizeZ = 1;
% nelx = ne;
% nely = ne;
% nelz = ne;
% Mesh = Mesh3D(sizeX, sizeY, sizeZ, nelx, nely, nelz);
fprintf('\nThe mesh was generated successfully with 8-node Brick (Hex8) element! \n');
% fprintf('The mesh consist of %d elements and %d nodes \a\n\n',Mesh.nel,Mesh.nnod);
% name = ['Mesh_',num2str(nelx),'x',num2str(nely),'x',num2str(nelz),'.mat'];
% save(name);
% 
% function Mesh = Mesh3D(sizeX, sizeY, sizeZ, nelx, nely, nelz)
% StructuredMesh3D: Construct a structured mesh over a parallelepiped with
%                   Brick elements with 8 nodes
% 
%
% INPUT
%           sizeX:          Size of the paralelepiped on X-direction
%           sizeY:          Size of the paralelepiped on Y-direction
%           sizeZ:          Size of the paralelepiped on Z-direction
%           nelx:           Number of elements on X-direction
%           nely:           Number of elements on Y-direction
%           nelz:           Number of elements on Z-direction
% 
% 
% OUTPUT
%           Mesh:           Data structure with the following: 
%               Mesh.ncoord:Coordinates (X,Y,Z)of the nodes
%               Mesh.conect:Conectivty matrix of the elements
%               Mesh.nel:   Number of total elements in the mesh
%               Mesh.nnod:  Number of total nodes in the mesh    
%               Mesh.nelx:  Number of elements on X-direction
%               Mesh.nely:  Number of elements on Y-direction
%               Mesh.nelz:  Number of elements on Z-direction
%               Mesh.sizeX: Size of the paralelepiped on X-direction
%               Mesh.sizeY: Size of the paralelepiped on Y-direction
%               Mesh.sizeZ: Size of the paralelepiped on Z-direction
%               Mesh.selX:  Size of the element on X-direction
%               Mesh.selY:  Size of the element on Y-direction
%               Mesh.selZ:  Size of the element on Z-direction
% 

%% NODAL COORDINATES
selX = sizeX/nelx;
selY = sizeY/nely;
selZ = sizeZ/nelz;
nx = uint32(nelx+1);
ny = uint32(nely+1);
nz = uint32(nelz+1);
[X,Y,Z]= ndgrid(selX*(0:nelx),selY*(0:nely),selZ*(0:nelz));
nodes = [X(:), Y(:), Z(:)];

%% CONECTIVITY OF THE ELEMENTS
elements = zeros(nelx*nely*nelz, 8, 'uint32');
el = 0;
for elz= 1:nelz
    for ely = 1:nely
        for elx = 1:nelx
            n1 = nx*ny*(elz-1)+nx*(ely-1)+elx;
            n4 = n1+nx;
            n5 = nx*ny*elz+nx*(ely-1)+elx;
            n8 = n5+nx;
            el = el + 1;
            elements(el,:)=[n1, n1+1, n4+1, n4, n5, n5+1, n8+1, n8];
        end
    end
end
