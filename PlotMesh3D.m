function PlotMesh3D(Mesh,pnodes,nelement)
% PlotMesh3D: Plot a structured grid in 3D with 8 node brick elements
%
% INPUT
%           Mesh:         Data structure with the following:
%               Mesh.ncoord:    Coordinates (X,Y,Z)of the nodes
%               Mesh.conect:    Conectivty matrix of the elements
%

fig = figure('color',[1 1 1]);
axh = axes('Parent',fig,'FontSize',14);
box(axh,'on');

%% PLOT THE ELEMENTS
n1 = Mesh.conect(1,:);
n2 = Mesh.conect(2,:);
n3 = Mesh.conect(3,:);
n4 = Mesh.conect(4,:);
n5 = Mesh.conect(5,:);
n6 = Mesh.conect(6,:);
n7 = Mesh.conect(7,:);
n8 = Mesh.conect(8,:);

Face1 = [n1' n2' n3' n4'];
Face2 = [n5' n6' n7' n8'];
Face3 = [n1' n2' n6' n5'];
Face4 = [n3' n4' n8' n7'];
Face5 = [n2' n3' n7' n6'];
Face6 = [n1' n4' n8' n5'];

Facesxy = [Face1;Face2];
Facesxz = [Face3;Face4];
Facesyz = [Face5;Face6];

Faces = [Facesxy;Facesxz;Facesyz];
patch('Vertices',Mesh.ncoord','Faces',Faces,'EdgeColor','k','FaceColor',[.8,.9,1]);

%% Graph configuration
xlabel(axh,'x','FontSize',17,'FontWeight','bold');
ylabel(axh,'y','FontSize',17,'FontWeight','bold');
zlabel(axh,'z','FontSize',17,'FontWeight','bold');
view(3);
axis equal;
axis tight;
alpha(0.5);

if pnodes == 1
    PlotNodeNumbers3D(Mesh)
end
if nelement == 1
    PlotElementNumbers3D(Mesh)
end


%%
function PlotNodeNumbers3D(Mesh)
hold on;
plot3(Mesh.ncoord(1,:),Mesh.ncoord(2,:),Mesh.ncoord(3,:),...
    'MarkerFaceColor',[0 0 0],'Marker','o','LineStyle','none');
% node = strcat('  N',num2str([1:length(Mesh.ncoord)]'));
node = strcat(num2str([1:length(Mesh.ncoord)]'));
text(Mesh.ncoord(1,:),Mesh.ncoord(2,:),Mesh.ncoord(3,:),node,'fontsize',8,'color','k');
hold off;


%%
function PlotElementNumbers3D(Mesh)
hold on;
for i = 1:length(Mesh.conect)
    X = Mesh.ncoord(1,Mesh.conect(:,i));
    Y = Mesh.ncoord(2,Mesh.conect(:,i));
    Z = Mesh.ncoord(3,Mesh.conect(:,i));
    text(sum(X)/8,sum(Y)/8,sum(Z)/8,int2str(i),'fontsize',10,'color','w','BackgroundColor',[0 0 0]);
end
hold off;
