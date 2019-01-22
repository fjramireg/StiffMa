function ke = Hex8vectorSym(X,D,L)
% HEX8VECTORSYM  Compute the lower symmetric part of the element stiffness
% matrix for a VECTOR problem taking advantage of simmetry.
%   HEX8VECTORSYM(X,D,L) returns the element stiffness matrix "ke" from
%   finite element analysis of vector problems in a three-dimensional
%   domain taking advantage of symmetry, where "X" is the nodal coordinates
%   of element "e", "D" the material property matrix for an isotropic
%   material, and "L" the shape function derivatives for the HEX8 elements.
%
%   See also ASSEMBLYVECTOR, HEX8VECTOR, HEX8VECTORSYMGPU
%
%   For more information, see <a href="matlab:
%   web('https://github.com/fjramireg/MatGen')">the MatGen Web site</a>.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created: 16/01/2019. Modified: 21/01/2019. Version: 1.3

B  = zeros(6,24,class(X));  % Initializes the matrix B
ke = zeros(300,1,class(X)); % Initializes the element stiffness matrix
for i=1:8                   % Loop over numerical integration points
    Li = L(:,:,i);          % Matrix L in point i
    Jac  = Li*X;            % Jacobian matrix
    detJ = det(Jac);        % Jacobian's determinant
    dNdX = Jac\Li;          % Shape function derivatives with respect to x,y,z
    % Matrix B
    B(1,1:3:24) = dNdX(1,:);
    B(2,2:3:24) = dNdX(2,:);
    B(3,3:3:24) = dNdX(3,:);
    B(4,1:3:24) = dNdX(2,:);  B(4,2:3:24) = dNdX(1,:);
    B(5,2:3:24) = dNdX(3,:);  B(5,3:3:24) = dNdX(2,:);
    B(6,1:3:24) = dNdX(3,:);  B(6,3:3:24) = dNdX(1,:);
    temp = 0;
    for j=1:24           % Loops to compute the symmetric part of ke
        for k=j:24
            idx = temp + k;
            ke(idx) = ke(idx) + (B(:,k)'*D*B(:,j))*detJ;
        end
        temp = temp + k - j;
    end
end
