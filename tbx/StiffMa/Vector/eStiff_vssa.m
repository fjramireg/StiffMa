function Ke = eStiffa_vss(Mesh, MP, sets)
% ESTIFFA_VSS Compute ALL (a) the element stiffness matrices for a VECTOR (v)
% problem by using a serial (s) code on the CPU taking advantage of simmetry.
%   ESTIFFA_VSS(Mesh, MP, sets) returns the element stiffness matrix "ke"
%   for all elements in a finite element analysis of vector problems in a
%   three-dimensional domain taking advantage of symmetry but in a serial
%   manner on the CPU,  where Mesh is a strucutre variable containing the
%   field "Mesh.elements" which is the connectivity matrix of size nelx8
%   and "Mesh.nodes" is the nodal coordinates array of size Nx3. MP is also
%   a struct with fields "MP.E" (Young's modulus) and "MP.nu" (Poisson
%   ratio) the material property for a linear isotropic material. The
%   struct "sets" must contain several similation parameters:
%   - sets.dTE is the data precision of "Mesh.elements"
%   - sets.dTN is the data precision of "Mesh.nodes"
%   - sets.nel is the number of finite elements
%   - sets.sz  is the umber of symmetry entries.
%
%   See also STIFFMA_VSS, ESTIFF_VSS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 28/01/2020. Version: 1.4. Name changed, Doc improved
% 	Modified: 28/01/2019. Version: 1.3
%   Created:  30/11/2018. Version: 1.0

L = dNdrst(sets.dTN);                    % Shape functions derivatives
D = DMatrix(MP.E, MP.nu, sets.dTN);      % Material matrix (isotropic)
Ke = zeros(sets.sz, sets.nel, sets.dTN); % Stores the NNZ values
for e = 1:sets.nel                       % Loop over elements
    n = Mesh.elements(e,:);              % Nodes of the element 'e'
    X = Mesh.nodes(n,:);                 % Nodal coordinates of the element 'e'
    Ke(:,e) = eStiff_vss(X,D,L,sets);    % Element stiffnes matrix storage
end
Ke = Ke(:);
