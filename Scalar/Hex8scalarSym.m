function ke = Hex8scalarSym(X,c,L)
% HEX8SCALARSYM Compute the lower symmetric part of the element stiffness
% matrix for a SCALAR problem taking advantage of simmetry.
%   HEX8SCALARSYM(X,c,L) returns the element stiffness matrix "ke" from
%   finite element analysis of scalar problems in a three-dimensional
%   domain taking advantage of symmetry, where "X" is the nodal coordinates
%   of element "e", "c" the material property for an isotropic material,
%   and "L" the shape function derivatives for the HEX8 elements.
%
%   See also STIFFMATGENSCSYMCPU, HEX8SCALAR, HEX8SCALARSYMCPU, HEX8SCALARSYMCPUp
%
%   For more information, see <a href="matlab:
%   web('https://github.com/fjramireg/MatGen')">the MatGen Web site</a>.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created: 30/11/2018. Modified: 21/01/2019. Version: 1.3

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
