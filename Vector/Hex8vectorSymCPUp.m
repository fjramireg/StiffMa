function Ke = Hex8vectorSymCPUp(elements,nodes,E,nu)
% HEX8VECTORSYMCPUP Compute the lower symmetric part of all the element
% stiffness matrices for a VECTOR problem taking advantage of simmetry and
% parallel computing on multicore CPU.
%   HEX8VECTORSYMCPUP(elements,nodes,E,nu) returns the element stiffness
%   matrix "ke" for all elements in a finite element analysis of vector
%   problems in a three-dimensional domain taking advantage of symmetry
%   computed in a parallel manner on multicore CPUs, where "elements" is
%   the connectivity matrix, "nodes" the nodal coordinates, and "E"
%   (Young's modulus) and "nu" (Poisson ratio) the material property for an
%   isotropic material. 
% 
%   See also STIFFMATGENVCSYMCPUP, HEX8VECTOR, HEX8VECTORSYM, HEX8VECTORSYMCPU
%
%   For more information, see <a href="matlab:
%   web('https://github.com/fjramireg/MatGen')">the MatGen Web site</a>.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created: 30/11/2018. Modified: 28/01/2019. Version: 1.3

dTN = class(nodes);                         % "nodes" data precision
L = dNdrst(dTN);                            % Shape functions derivatives
nel = size(elements,1);                     % Total number of elements
D = MaterialMatrix(E,nu,dTN);               % Material matrix (isotropic)
Ke = zeros(300,nel,dTN);                    % Stores the NNZ values
Xg = reshape(nodes(elements',:)',[3,8,nel]);% Nodal coordinates reorganized to parfor
parfor e = 1:nel                            % Loop over elements
    Ke(:,e) = Hex8vectorSym(Xg(:,:,e)',D,L);% Element stiffnes matrix storage
end
Ke = Ke(:);
