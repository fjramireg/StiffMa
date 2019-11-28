function Ke = Hex8scalarSymCPU(elements,nodes,c)
% HEX8SCALARSYMCPU Compute the lower symmetric part of all the element
% stiffness matrices for a SCALAR problem taking advantage of simmetry on
% the CPU by using a serial code.
%   HEX8SCALARSYMCPU(elements,nodes,c) returns the element stiffness
%   matrix "ke" for all elements in a finite element analysis of scalar
%   problems in a three-dimensional domain taking advantage of symmetry but
%   in a serial manner on the CPU,  where "elements" is the connectivity
%   matrix, "nodes" the nodal coordinates, and "c" the material property
%   for an isotropic material.
%
%   See also STIFFMATGENSCSYMCPU, HEX8SCALAR, HEX8SCALARSYM, HEX8SCALARSYMGPU
%
%   For more information, see <a href="matlab:
%   web('https://github.com/fjramireg/MatGen')">the MatGen Web site</a>.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created: 30/11/2018. Modified: 22/01/2019. Version: 1.3

L = dNdrst(class(nodes));           % Shape functions derivatives
nel = size(elements,1);             % Total number of elements
Ke = zeros(36, nel, class(nodes));  % Stores the NNZ values
for e = 1:nel                       % Loop over elements
    n = elements(e,:);              % Nodes of the element 'e'
    X = nodes(n,:);                 % Nodal coordinates of the element 'e'
    Ke(:,e) = Hex8scalarSym(X,c,L); % Symmetric part of ke
end
Ke = Ke(:);
