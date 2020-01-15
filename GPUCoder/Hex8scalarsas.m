function Ke = Hex8scalarsas(elements,nodes,c)  %#codegen
% HEX8SCALARSAS Compute the lower symmetric part of all ke in SERIAL computing
% for a SCALAR problem on the CPU.
%   HEX8SCALARSAS(elements,nodes,c) returns the element stiffness matrix "ke"
%   for all elements in a finite element analysis of scalar problems in a
%   three-dimensional domain taking advantage of symmetry but in a serial manner
%   on the CPU,  where "elements" is the connectivity matrix of size nelx8,
%   "nodes" the nodal coordinates of size Nx3, and "c" the material property for
%   an isotropic material (scalar). 
%
%   See also STIFFMASS, HEX8SCALARS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 07/12/2019. Version: 1.4. Name changed, Doc improved
% 	Modified: 22/01/2019. Version: 1.3
%   Created:  30/11/2018. Version: 1.0

% Add kernelfun pragma to trigger kernel creation
coder.gpu.kernelfun;

L = dNdrst(class(nodes));           % Shape functions derivatives
nel = size(elements,1);             % Total number of elements
Ke = zeros(36, nel, class(nodes));  % Stores the NNZ values
for e = 1:nel                       % Loop over elements
    n = elements(e,:);              % Nodes of the element 'e'
    X = nodes(n,:);                 % Nodal coordinates of the element 'e'
    Ke(:,e) = Hex8scalarss(X,c,L);  % Symmetric part of ke
end
Ke = Ke(:);
