function ke = eStiff_vss(X,D,L,sets)
% ESTIFF_VSS  Compute the element stiffness matrix for a VECTOR (v) problem in
% SERIAL (s) computing taking advantage of simmetry (s).
%   ESTIFF_VSS(X,D,L) returns the element stiffness matrix "ke" from finite
%   element analysis of vector problems in a three-dimensional domain
%   taking advantage of symmetry, where "X" is the nodal coordinates of
%   element "e", "D" the material property matrix for an isotropic
%   material, and "L" the shape function derivatives for the HEX8 elements.
%   The  struct "sets" must contain several similation parameters:
%   - sets.dTN is the data precision of "Mesh.nodes"
%   - sets.edof is the number of DOFs per element
%   - sets.sz  is the umber of symmetry entries.
%
%   See also ESTIFFA_VSS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 28/01/2020. Version: 1.4. Name changed, Doc improved
% 	Modified: 21/01/2019. Version: 1.3
%   Created:  16/01/2019. Version: 1.0

B  = zeros(6, sets.edof, sets.dTN);	% Initializes the matrix B
ke = zeros(sets.sz, 1, sets.dTN);   % Initializes the element stiffness matrix
for i=1:8                           % Loop over numerical integration points
    Li = L(:,:,i);                  % Matrix L in point i
    Jac  = Li*X;                    % Jacobian matrix
    detJ = det(Jac);                % Jacobian's determinant
    dNdX = Jac\Li;                  % Shape function derivatives with respect to x,y,z
    B(1,1:3:24) = dNdX(1,:);        % Matrix B filling
    B(2,2:3:24) = dNdX(2,:);
    B(3,3:3:24) = dNdX(3,:);
    B(4,1:3:24) = dNdX(2,:);  B(4,2:3:24) = dNdX(1,:);
    B(5,2:3:24) = dNdX(3,:);  B(5,3:3:24) = dNdX(2,:);
    B(6,1:3:24) = dNdX(3,:);  B(6,3:3:24) = dNdX(1,:);
    temp = 0;
    for j=1:24                      % Loops to compute the symmetric part of ke
        for k=j:24
            idx = temp + k;
            ke(idx) = ke(idx) + (B(:,k)'*D*B(:,j))*detJ;
        end
        temp = temp + k - j;
    end
end
