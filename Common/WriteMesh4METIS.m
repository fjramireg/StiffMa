function WriteMesh4METIS(elements)
% WRITEMESH4METIS writes a file that is able to be read by METIS for mesh
% partitioning.
%
%  INPUT:
%   elements:           Conectivty matrix of the elements [nelx8]
%
%  OUTPUT:
%   file:               Writes a file named "metis.mesh"
%
%   See also CREATEMESH.
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  11/04/2020. Version: 1.0

nel = size(elements,1);
metismesh = fopen('metis.mesh','w');
fprintf(metismesh,'%u\n',nel);
% Export elements
for e=1:nel
    fprintf(metismesh,'%u %u %u %u %u %u %u %u\n',elements(e,1),elements(e,2),...
        elements(e,3),elements(e,4),elements(e,5),elements(e,6),...
        elements(e,7),elements(e,8));
end
fclose(metismesh);