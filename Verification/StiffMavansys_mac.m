function StiffMavansys_mac(elements,nodes,E,nu)
% STIFFMAVANSYS_MAC Create an APDL macro to export the global stiffness matrix
% tril(K) for a VECTOR problem in ANSYS.
%   STIFFMAVANSYS_MAC(elements,nodes,E,nu) Create an APDL macro to export the
%   global stiffness matrix from ANSYS, where "elements" is the connectivity
%   matrix of size nelx8, "nodes" the nodal coordinates of size Nx3, and
%   "E"/"nu" the material property for an isotropic material (scalars values). 
%
%   See also STIFFMAV
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  18/12/2019. Version: 1.0

afile = fopen('StiffMavansys.mac','w');
fprintf(afile,'/NOPR\n');
fprintf(afile,'/FILNAME,STIFFNESS_VEC,0\n');
fprintf(afile,'/TITLE,STIFFNESS_VEC\n');
fprintf(afile,'/PREP7\n');
fprintf(afile,'ET,1,SOLID185\n');   % SOLID185 is used for 3-D modeling of solid structures. It is defined by eight nodes having three degrees of freedom at each node: translations in the nodal x, y, and z directions.
fprintf(afile,'MP,EX,1,%E\n',E);    % Elastic moduli
fprintf(afile,'MP,PRXY,1,%E\n',nu); % Major Poisson's ratio

% Export nodes
for n=1:size(nodes,1)
    fprintf(afile,'N,%u,%E,%E,%E\n',n,nodes(n,1),nodes(n,2),nodes(n,3));
end
fprintf(afile,'NWRITE\n');

% Export elements
for e=1:size(elements,1)
    fprintf(afile,'E,%u,%u,%u,%u,%u,%u,%u,%u\n',elements(e,1),elements(e,2),...
        elements(e,3),elements(e,4),elements(e,5),elements(e,6),elements(e,7),elements(e,8));
end
fprintf(afile,'EWRITE\n');

% Stops solution after assembling global matrices.
fprintf(afile,'FINISH\n');
fprintf(afile,'/SOL\n');
fprintf(afile,'EMATWRITE,YES\n');
fprintf(afile,'WRFULL,1\n');
fprintf(afile,'SOLVE\n');
fprintf(afile,'FINISH\n');

% Export stiffness matrix. Opt1: Export the matrix in the Harwell-Boeing file format
fprintf(afile,'/AUX2\n');
fprintf(afile,"FILE,'STIFFNESS_VEC','full',' '\n");
fprintf(afile,"HBMAT,'STIFF_ANSYS','hb',' ',ASCII,STIFF,NO,YES\n"); % Writes an assembled global matrix in Harwell-Boeing format.
% HBMAT, Fname, Ext, --, Form, Matrx, Rhs, Mapping
% The mapping file can be used to map the matrix equation numbers found in the .MATRIX file directly to the corresponding node numbers and degrees of freedom.
fprintf(afile,'FINISH\n');

% Export stiffness matrix. Opt2: Export the matrix in the Matrix Market Format.
fprintf(afile,"*SMAT,STIFFMAT,D,IMPORT,FULL,'STIFFNESS_VEC.full',STIFF\n");
fprintf(afile,"*EXPORT,STIFFMAT,MMF,'STIFF_ANSYS.mmf'\n");

% Export element stiffness matrix in the Matrix Market Format.
for e=1:size(elements,1)
    fprintf(afile,"*DMAT,KE%s,D,IMPORT,EMAT,'STIFFNESS_VEC.emat',STIFF,%u\n",num2str(e),e);
    fprintf(afile,"*EXPORT,KE%s,MMF,'KE%s.dat'\n",num2str(e),num2str(e));
end

% End
fprintf(afile,'FINISH\n');
fprintf(afile,'/GO\n');
fclose(afile);
