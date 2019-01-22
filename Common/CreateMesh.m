function [elements, nodes] = CreateMesh(nelx,nely,nelz,dTypeE,dTypeN,PlotE,PlotN)
% Simple mesher of a unit cubic with configurable and structured discretization [Hex8]
%  INPUT:
%   nelx, nely, nelz:   Number of elements on X-Y-Z direction
%   dTypeN:             Data precision for "nodes" ['single' or 'double']
%   dTypeE:             Data precision for "elements" ['uint32', 'uint64' or 'double']
%   PlotE:              Plot the elements and their numbers (1 to plot)
%   PlotN:              Plot the nodes and their numbers (1 to plot)
%
%  OUTPUT:
%   elements:           Conectivty matrix of the elements [nelx8]
%   nodes:              X,Y,Z coordinates of the nodes [nnodx3]
%
%   See also ASSEMBLYSCALAR, ASSEMBLYVECTOR
%
%   For more information, see <a href="matlab:
%   web('https://github.com/fjramireg/MatGen')">the MatGen Web site</a>.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created: 30/11/2018. Modified: 21/01/2019. Version: 1.3

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
elseif ~( isscalar(PlotE) || isscalar(PlotN) )
    error('Error. Input "PlotE" and "PlotN" must be SCALAR (1 to plot, or other to not)');
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
fprintf('\nThe mesh was created successfully with %u Hex8 elements and %u nodes! \n\n\n',...
    el,size(nodes,1));

%% PLOTS
if ( PlotE==1 || PlotN==1 )
    fig = figure('color',[1 1 1]); axh = axes('Parent',fig,'FontSize',14); box(axh,'on');
    
    % Plot the elements
    n1 = elements(:,1); n2 = elements(:,2); n3 = elements(:,3); n4 = elements(:,4);
    n5 = elements(:,5); n6 = elements(:,6); n7 = elements(:,7); n8 = elements(:,8);
    Face1 = [n1 n2 n3 n4]; Face2 = [n5 n6 n7 n8]; Face3 = [n1 n2 n6 n5];
    Face4 = [n3 n4 n8 n7]; Face5 = [n2 n3 n7 n6]; Face6 = [n1 n4 n8 n5];
    Facesxy = [Face1;Face2]; Facesxz = [Face3;Face4]; Facesyz = [Face5;Face6];
    Faces = [Facesxy;Facesxz;Facesyz];
    patch('Vertices',nodes,'Faces',Faces,'EdgeColor','k','FaceColor',[.8,.9,1]);
    
    % Graph configuration
    xlabel(axh,'x','FontSize',17,'FontWeight','bold');
    ylabel(axh,'y','FontSize',17,'FontWeight','bold');
    zlabel(axh,'z','FontSize',17,'FontWeight','bold');
    view(3); axis equal; axis tight; alpha(0.3); hold on;
    
    if PlotN == 1       % Plot the nodes and their numbers
        plot3(nodes(:,1),nodes(:,2),nodes(:,3),'MarkerFaceColor',[0 0 0],'Marker','o','LineStyle','none');
        nm = strcat('  N',num2str([1:length(nodes)]'));
        text(nodes(:,1),nodes(:,2),nodes(:,3),nm,'fontsize',8,'color','k');
    end
    if PlotE == 1    % Plot the elements numbering
        for i = 1:el
            X = nodes(elements(i,:),1); Y = nodes(elements(i,:),2); Z = nodes(elements(i,:),3);
            text(sum(X)/8,sum(Y)/8,sum(Z)/8,int2str(i),'fontsize',10,'color','w','BackgroundColor',[0 0 0]);
        end
    end
    hold off;
end
