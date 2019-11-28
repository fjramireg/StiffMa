function Ke = Hex8scalarSymCPUp(elements,nodes,c)
% HEX8SCALARSYMCPUP Compute the lower symmetric part of all the element
% stiffness matrices for a SCALAR problem taking advantage of simmetry and
% multicore CPU by using parallel computing.
%   HEX8SCALARSYMCPUP(elements,nodes,c) returns the element stiffness
%   matrix "ke" for all elements in a finite element analysis of scalar
%   problems in a three-dimensional domain taking advantage of symmetry
%   computed in a parallel manner on multicore CPUs,  where "elements" is
%   the connectivity matrix, "nodes" the nodal coordinates, and "c" the
%   material property for an isotropic material.
%
%   See also STIFFMATGENSCSYMCPUP, HEX8SCALAR, HEX8SCALARSYM, HEX8SCALARSYMCPU
%
%   For more information, see <a href="matlab:
%   web('https://github.com/fjramireg/MatGen')">the MatGen Web site</a>.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created: 28/01/2019. Modified: 28/01/2019. Version: 1.3

L  = dNdrst(class(nodes));                  % Shape functions derivatives
nel= size(elements,1);                      % Total number of elements
Ke = zeros(36, nel, class(nodes));          % Stores the NNZ values
Xg = reshape(nodes(elements',:)',[3,8,nel]);% Nodal coordinates reorganized to parfor
parfor e = 1:nel                            % Loop over elements
    Ke(:,e) = Hex8scalarSym(Xg(:,:,e)',c,L);% Symmetric part of ke
end
Ke = Ke(:);
