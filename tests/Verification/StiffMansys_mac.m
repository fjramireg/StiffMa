function StiffMansys_mac(elements,nodes,c)
% STIFFMANSYS_MAC Create an APDL macro to export the global stiffness matrix
% tril(K) for a SCALAR problem in ANSYS.
%   STIFFMANSYS_MAC(elements,nodes,c) Create an APDL macro to export the global
%   stiffness matrix from ANSYS, where "elements" is the connectivity matrix of
%   size nelx8, "nodes" the nodal coordinates of size Nx3, and "c" the material
%   property for an isotropic material (scalar). 
%
%   See also STIFFMAS, STIFFMAPS, SPARSE, ACCUMARRAY
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  14/12/2019. Version: 1.0

afile = fopen('StiffMansys.mac','w');
fprintf(afile,'/NOPR\n');
fprintf(afile,'/FILNAME,STIFFNESS_MAT,0\n');
fprintf(afile,'/TITLE,STIFFNESS_MAT\n');
fprintf(afile,'/PREP7\n');
fprintf(afile,'ET,1,SOLID278\n');
fprintf(afile,'MP,KXX,1,%E\n',c);

% Export nodes
for n=1:size(nodes,1)
    fprintf(afile,'N,%u,%E,%E,%E\n',n,nodes(n,1),nodes(n,2),nodes(n,3));
end

% Export elements
for e=1:size(elements,1)
    fprintf(afile,'E,%u,%u,%u,%u,%u,%u,%u,%u\n',elements(e,1),elements(e,2),...
        elements(e,3),elements(e,4),elements(e,5),elements(e,6),elements(e,7),elements(e,8));
end

% Stops solution after assembling global matrices.
fprintf(afile,'FINISH\n');
fprintf(afile,'/SOL\n');
fprintf(afile,'EMATWRITE,YES\n');
fprintf(afile,'WRFULL,1\n');
fprintf(afile,'SOLVE\n');
fprintf(afile,'FINISH\n');

% Export stiffness matrix. Opt1: Export the matrix in the Harwell-Boeing file format
fprintf(afile,'/AUX2\n');
fprintf(afile,"FILE,'STIFFNESS_MAT','full',' '\n");
fprintf(afile,"HBMAT,'STIFF_ANSYS','HB',' ',ASCII,STIFF,NO,YES\n"); % Writes an assembled global matrix in Harwell-Boeing format.
% HBMAT, Fname, Ext, --, Form, Matrx, Rhs, Mapping
% The mapping file can be used to map the matrix equation numbers found in the .MATRIX file directly to the corresponding node numbers and degrees of freedom.
fprintf(afile,'FINISH\n');

% Export stiffness matrix. Opt2: Export the matrix in the Matrix Market Format.
fprintf(afile,"*SMAT,STIFFMAT,D,IMPORT,FULL,'STIFFNESS_MAT.full',STIFF\n");
fprintf(afile,"*EXPORT,STIFFMAT,MMF,'STIFF_ANSYS.mmf'\n");

% Export element stiffness matrix in the Matrix Market Format.
for e=1:size(elements,1)
    fprintf(afile,"*DMAT,KE%s,D,IMPORT,EMAT,'STIFFNESS_MAT.emat',STIFF,%u\n",num2str(e),e);
    fprintf(afile,"*EXPORT,KE%s,MMF,'KE%s.dat'\n",num2str(e),num2str(e));
end

% End
fprintf(afile,'FINISH\n');
fprintf(afile,'/GO\n');
fclose(afile);
