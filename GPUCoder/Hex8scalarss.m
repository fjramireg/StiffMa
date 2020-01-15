function ke = Hex8scalarss(X,c,L)
% HEX8SCALARSS Compute the lower symmetric part of the element stiffness matrix
% for a SCALAR problem in SERIAL computing on CPU.
%   HEX8SCALARSS(X,c,L) returns the element stiffness matrix "ke" from finite
%   element analysis of scalar problems in a three-dimensional domain taking
%   advantage of symmetry, where "X" is the nodal coordinates of element "e" of
%   size 8x3, "c" the material property for an isotropic material (scalar), and
%   "L" the shape function derivatives for the HEX8 elements of size 3x3x8. 
%
%   See also HEX8SCALARSAS, HEX8SCALARS, HEX8SCALARSAP
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 07/12/2019. Version: 1.4. Name changed, Doc improved
% 	Modified: 21/01/2019. Version: 1.3
%   Created:  30/11/2018. Version: 1.0

ke = zeros(36,1,class(X));% Initializes the element stiffness matrix
for i=1:8               % Loop over numerical integration
    Li = L(:,:,i);      % Matrix L in point i
    Jac  = Li*X;        % Jacobian matrix
    detJ = det(Jac);    % Jacobian's determinant
    B = Jac\Li;         % B matrix
    temp = 0;
    for j=1:8           % Loops to compute the symmetric part of ke
        for k=j:8
            idx = temp + k;
            ke(idx) = ke(idx) + c*detJ*(B(:,k)'*B(:,j));
        end
        temp = temp + k - j;
    end
end
