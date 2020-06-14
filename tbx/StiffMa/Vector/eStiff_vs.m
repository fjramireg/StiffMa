function ke = eStiff_vs(X,D,dType)
% ESTIFF_VS Compute the element stiffness matrix for a VECTOR (s) problem
% in SERIAL computing (s). 
%   ESTIFF_VS(X,D) returns the element stiffness matrix "ke" for an element
%   "e" in a finite element analysis of vector problems in a three-
%   dimensional domain computed in a serial manner on the CPU,  where "X"
%   is the nodal coordinates of the element "e" (size 8x3), and "D" the 
%   material property matrix of size 6x6. As input it requieres the data
%   type "dType" as  single or double.
%
%   See also STIFFMA_VS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 28/01/2020. Version: 1.4. Name changed, Doc improved
% 	Modified: 21/01/2019. Version: 1.3
%   Created:  16/01/2019. Version: 1.0

p = zeros(1,dType);         % Initialize p in the correct data type
p(1) = 1/sqrt(3);           % Gauss points value
r = [-p,p,p,-p,-p,p,p,-p];  % Points through r-coordinate
s = [-p,-p,p,p,-p,-p,p,p];  % Points through s-coordinate
t = [-p,-p,-p,-p,p,p,p,p];  % Points through t-coordinate
ke = zeros(24,24);          % Initialize the element stiffness matrix
B  = zeros(6,24);           % Initializes the matrix B
for i=1:8                   % Loop over numerical integration
    ri = r(i); si = s(i); ti = t(i);
    a = 1-ri; b = 1+ri; c = 1-si; d = 1+si; e = 1-ti; f = 1+ti;
    L = [-c*e, c*e, d*e, -d*e, -c*f, c*f, d*f, -d*f;...     % dN/dr
        -a*e, -b*e, b*e, a*e, -a*f, -b*f, b*f, a*f;...      % dN/ds
        -a*c, -b*c, -b*d, -a*d, a*c, b*c, b*d, a*d]/8;      % dN/dt
    Jac  = L*X;             % Jacobian matrix
    detJ = det(Jac);        % Jacobian's determinant
    dNdX = Jac\L;           % Shape function derivatives with respect to x,y,z
    B(1,1:3:24) = dNdX(1,:);% Matrix B fill
    B(2,2:3:24) = dNdX(2,:);
    B(3,3:3:24) = dNdX(3,:);
    B(4,1:3:24) = dNdX(2,:);  B(4,2:3:24) = dNdX(1,:);
    B(5,2:3:24) = dNdX(3,:);  B(5,3:3:24) = dNdX(2,:);
    B(6,1:3:24) = dNdX(3,:);  B(6,3:3:24) = dNdX(1,:);
    ke = ke + B'*D*B*detJ;  % Element stiffness matrix
end
