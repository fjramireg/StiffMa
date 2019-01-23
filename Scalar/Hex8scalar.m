function ke = Hex8scalar(X,c)
% HEX8SCALAR Compute the element stiffness matrix for a SCALAR problem.
%   HEX8SCALAR(X,c) returns the element stiffness matrix "ke" from finite
%   element analysis of scalar problems in a three-dimensional domain
%   where "X" is the nodal coordinates of element "e" and "c" the material
%   property for an isotropic material.
%
%   See also STIFFMATGENSC, STIFFMATGENSCSYM, HEX8SCALARSYM, HEX8SCALARSYMGPU
%
%   For more information, see <a href="matlab:
%   web('https://github.com/fjramireg/MatGen')">the MatGen Web site</a>.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created: 30/11/2018. Modified: 21/01/2019. Version: 1.3

p = 1/sqrt(3);                  % Gauss points
r = [p,-p,p,-p,p,-p,p,-p];      % Points through r-coordinate
s = [p,p,-p,-p,p,p,-p,-p];      % Points through s-coordinate
t = [p,p,p,p,-p,-p,-p,-p];      % Points through t-coordinate
ke = zeros(8,8);                % Initializes the element stiffness matrix
for i=1:8                       % Loop over numerical integration
    ri = r(i); si = s(i); ti = t(i);
    %  Shape function derivatives with respect to r,s,t
    dNdr = (1/8)*[-(1-si)*(1-ti),  (1-si)*(1-ti), (1+si)*(1-ti), -(1+si)*(1-ti),...
        -(1-si)*(1+ti),  (1-si)*(1+ti), (1+si)*(1+ti), -(1+si)*(1+ti)];
    dNds = (1/8)*[-(1-ri)*(1-ti), -(1+ri)*(1-ti), (1+ri)*(1-ti),  (1-ri)*(1-ti),...
        -(1-ri)*(1+ti), -(1+ri)*(1+ti), (1+ri)*(1+ti),  (1-ri)*(1+ti)];
    dNdt = (1/8)*[-(1-ri)*(1-si), -(1+ri)*(1-si),-(1+ri)*(1+si), -(1-ri)*(1+si),...
        (1-ri)*(1-si),  (1+ri)*(1-si), (1+ri)*(1+si),  (1-ri)*(1+si)];
    L = [dNdr; dNds; dNdt];     % L matrix
    Jac  = L*X;                 % Jacobian matrix
    detJ = det(Jac);            % Jacobian's determinant
    B = Jac\L;                  % B matrix
    ke = ke + c*detJ*(B'*B);    % Element stiffness matrix
end
