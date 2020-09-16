function ke = eStiff_ss(X,ct,dTN)
% ESTIFF_SS Computes the element stiffness matrix for a SCALAR (s) problem
% in SERIAL (s) computing.
%   ke = ESTIFF_SS(X,c,dTN) returns the element stiffness matrix "ke" for
%   an element "e" in a finite element analysis of scalar problems in a
%   three-dimensional domain computed in a serial manner on the CPU,  where
%   "X" is the nodal coordinates of the element "e" (size 8x3), and "ct" the
%   material property (scalar). dTN is the data type for nodal coordinates.
%
%   Example:
%         X = [-1,-1,-1; 1,-1,-1; 1,1,-1; -1,1,-1; -1,-1,1; 1,-1,1; 1,1,1; -1,1,1]
%         ke = eStiff_ss(X,1,'single')
%
%   See also STIFFMA_SS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 05/12/2019. Version: 1.4. Name changed, Doc improved
% 	Modified: 22/01/2019. Version: 1.3
%   Created:  30/11/2018. Version: 1.0

p = zeros(1,dTN);           % Initialize p in the correct data type
p(1) = 1/sqrt(3);           % Gauss points value
r = [-p,p,p,-p,-p,p,p,-p];  % Points through r-coordinate
s = [-p,-p,p,p,-p,-p,p,p];  % Points through s-coordinate
t = [-p,-p,-p,-p,p,p,p,p];  % Points through t-coordinate
ke = zeros(8,8,dTN);        % Initialize the element stiffness matrix
for i=1:8                   % Loop over numerical integration
    ri = r(i); si = s(i); ti = t(i);
    a = 1-ri; b = 1+ri; c = 1-si; d = 1+si; e = 1-ti; f = 1+ti;
    L = [-c*e, c*e, d*e, -d*e, -c*f, c*f, d*f, -d*f;...     % dN/dr
        -a*e, -b*e, b*e, a*e, -a*f, -b*f, b*f, a*f;...      % dN/ds
        -a*c, -b*c, -b*d, -a*d, a*c, b*c, b*d, a*d]/8;      % dN/dt
    Jac  = L*X;             % Jacobian matrix
    detJ = det(Jac);        % Jacobian's determinant
    B = Jac\L;              % B matrix
    ke = ke + ct*detJ*(B'*B);% Element stiffness matrix
end
