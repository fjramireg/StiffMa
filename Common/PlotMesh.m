function PlotMesh(elements, nodes, PlotE, PlotN)
% PLOTMESH plots the domain with its discretization using Hex8 elements
%  INPUT:
%   elements:           Conectivty matrix of the elements [nelx8]
%   nodes:              X,Y,Z coordinates of the nodes [Nx3]
%   nelx, nely, nelz:   Number of elements on X-Y-Z direction
%   PlotE:              Plot the element numbers
%   PlotN:              Plot the node numbers
%
%  OUTPUT:              A graph of the mesh
%
%   See also CREATEMESH
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  14/12/2018. Version: 1.0

% INPUTS CHECK
if ~( isscalar(PlotE) || isscalar(PlotN) )
    error('Error. Input "PlotE" and "PlotN" must be SCALAR (1 to plot, or other to not)');
end

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
% xlabel(axh,'x','FontSize',17,'FontWeight','bold');
% ylabel(axh,'y','FontSize',17,'FontWeight','bold');
% zlabel(axh,'z','FontSize',17,'FontWeight','bold');
view(3); axis equal; axis tight; alpha(0.3); hold on;

if PlotN == 1       % Plot the nodes and their numbers
    plot3(nodes(:,1),nodes(:,2),nodes(:,3),'MarkerFaceColor',[0 0 0],'Marker','o','LineStyle','none');
    text(nodes(:,1),nodes(:,2),nodes(:,3),num2str([1:length(nodes)]'),'fontsize',8,'color','k');
end
if PlotE == 1    % Plot the elements numbering
    for i = 1:size(elements,1)
        X = nodes(elements(i,:),1); Y = nodes(elements(i,:),2); Z = nodes(elements(i,:),3);
        text(sum(X)/8,sum(Y)/8,sum(Z)/8,int2str(i),'fontsize',10,'color','w','BackgroundColor',[0 0 0]);
    end
end
hold off;
