function Ke = Hex8vectorSymCPU(elements,nodes,E,nu)
% HEX8VECTORSYMCPU Compute the lower symmetric part of all the element
% stiffness matrices for a VECTOR problem taking advantage of simmetry on
% the CPU by using a serial code.
%   HEX8VECTORSYMCPU(elements,nodes,E,nu) returns the element stiffness
%   matrix "ke" for all elements in a finite element analysis of vector
%   problems in a three-dimensional domain taking advantage of symmetry but
%   in a serial manner on the CPU,  where "elements" is the connectivity
%   matrix, "nodes" the nodal coordinates, and "E" (Young's modulus) and
%   "nu" (Poisson ratio) the material property for an isotropic material.
%
%   See also STIFFMATGENVCSYMCPU, HEX8VECTOR, HEX8VECTORSYM, HEX8VECTORSYMGPU
%
%   For more information, see <a href="matlab:
%   web('https://github.com/fjramireg/MatGen')">the MatGen Web site</a>.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created: 30/11/2018. Modified: 28/01/2019. Version: 1.3

dTN = class(nodes);                 % "nodes" data precision
L = dNdrst(dTN);                    % Shape functions derivatives
nel = size(elements,1);             % Total number of elements
D = MaterialMatrix(E,nu,dTN);       % Material matrix (isotropic)
Ke = zeros(300,nel,dTN);            % Stores the NNZ values
for e = 1:nel                       % Loop over elements
    n = elements(e,:);              % Nodes of the element 'e'
    X = nodes(n,:);                 % Nodal coordinates of the element 'e'
    Ke(:,e) = Hex8vectorSym(X,D,L); % Element stiffnes matrix storage
end
Ke = Ke(:);
