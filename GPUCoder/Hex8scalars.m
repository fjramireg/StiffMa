function ke = Hex8scalars(X,c) %#codegen
% HEX8SCALARS Compute the element stiffnes matrix for a SCALAR problem in SERIAL computing.
%   HEX8SCALARS(X,c) returns the element stiffness matrix "ke" for an element
%   "e"  in a finite element analysis of scalar problems in a three-dimensional
%   domain computed in a serial manner on the CPU,  where "X" is the nodal
%   coordinates of the element "e" (size 8x3), and "c" the material property
%   (scalar).
%
%   Examples:
%         X = [-1,-1,-1; 1,-1,-1; 1,1,-1; -1,1,-1; -1,-1,1; 1,-1,1; 1,1,1; -1,1,1]
%         ke = Hex8scalars(X,1)
% 
%   See also HEX8SCALARSAS, HEX8SCALARSAP
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 05/12/2019. Version: 1.4. Name changed, Doc improved
% 	Modified: 22/01/2019. Version: 1.3
%   Created:  30/11/2018. Version: 1.0

p = 1/sqrt(3);              % Gauss points
r = [-p,p,p,-p,-p,p,p,-p];  % Points through r-coordinate
s = [-p,-p,p,p,-p,-p,p,p];  % Points through s-coordinate
t = [-p,-p,-p,-p,p,p,p,p];  % Points through t-coordinate
ke = zeros(8,8);            % Initialize the element stiffness matrix
for i=1:8                   % Loop over numerical integration
    ri = r(i); si = s(i); ti = t(i);
    %  Shape function derivatives with respect to r,s,t
    dNdr = (1/8)*[-(1-si)*(1-ti),  (1-si)*(1-ti), (1+si)*(1-ti), -(1+si)*(1-ti),...
        -(1-si)*(1+ti),  (1-si)*(1+ti), (1+si)*(1+ti), -(1+si)*(1+ti)];
    dNds = (1/8)*[-(1-ri)*(1-ti), -(1+ri)*(1-ti), (1+ri)*(1-ti),  (1-ri)*(1-ti),...
        -(1-ri)*(1+ti), -(1+ri)*(1+ti), (1+ri)*(1+ti),  (1-ri)*(1+ti)];
    dNdt = (1/8)*[-(1-ri)*(1-si), -(1+ri)*(1-si),-(1+ri)*(1+si), -(1-ri)*(1+si),...
        (1-ri)*(1-si),  (1+ri)*(1-si), (1+ri)*(1+si),  (1-ri)*(1+si)];
    L = [dNdr; dNds; dNdt]; % L matrix
    Jac  = L*X;             % Jacobian matrix
    detJ = det(Jac);        % Jacobian's determinant
    B = Jac\L;              % B matrix
    ke = ke + c*detJ*(B'*B);% Element stiffness matrix
end
