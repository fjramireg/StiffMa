%  * ====================================================================*/
% ** This function was developed by:
%  *          Francisco Javier Ramirez-Gil
%  *          Universidad Nacional de Colombia - Medellin
%  *          Department of Mechanical Engineering
%  *
%  ** Please cite this code as:
%  *
%  ** Date & version
%  *      16/01/2019.
%  *      V 1.2
%  *
%  * ====================================================================*/

function ke = Hex8vector2(X,D)
% Element stiffness matrix ke for a VECTOR problem
p = 1/sqrt(3);              % Gauss points
r = [p,-p,p,-p,p,-p,p,-p];  % Points through r-coordinate
s = [p,p,-p,-p,p,p,-p,-p];  % Points through s-coordinate
t = [p,p,p,p,-p,-p,-p,-p];  % Points through t-coordinate
ke = zeros(24,24);          % Initialize the element stiffness matrix
B  = zeros(6,24);           % Initializes the matrix B
for i=1:8                   % Loop over numerical integration
    ri = r(i); si = s(i); ti = t(i);
    %  Shape function derivatives with respect to r,s,t
    dNdr = (1/8)*[-(1-si)*(1-ti),  (1-si)*(1-ti), (1+si)*(1-ti), -(1+si)*(1-ti),...
        -(1-si)*(1+ti),  (1-si)*(1+ti), (1+si)*(1+ti), -(1+si)*(1+ti)];
    dNds = (1/8)*[-(1-ri)*(1-ti), -(1+ri)*(1-ti), (1+ri)*(1-ti),  (1-ri)*(1-ti),...
        -(1-ri)*(1+ti), -(1+ri)*(1+ti), (1+ri)*(1+ti),  (1-ri)*(1+ti)];
    dNdt = (1/8)*[-(1-ri)*(1-si), -(1+ri)*(1-si),-(1+ri)*(1+si), -(1-ri)*(1+si),...
        (1-ri)*(1-si),  (1+ri)*(1-si), (1+ri)*(1+si),  (1-ri)*(1+si)];

    % Jacobian
    Jacobian = [dNdr ; dNds; dNdt]*X;
    detJ = det(Jacobian);
    R = inv(Jacobian);
    
    %  Shape Function Derivates with respect to x,y,z
    dNdx = R(1,1)*dNdr + R(1,2)*dNds + R(1,3)*dNdt;
    dNdy = R(2,1)*dNdr + R(2,2)*dNds + R(2,3)*dNdt;
    dNdz = R(3,1)*dNdr + R(3,2)*dNds + R(3,3)*dNdt;
    
    % Gradient matrix
    B(1,1:3:24) = dNdx;
    B(2,2:3:24) = dNdy;
    B(3,3:3:24) = dNdz;
    B(4,1:3:24) = dNdy; B(4,2:3:24) = dNdx;
    B(5,2:3:24) = dNdz; B(5,3:3:24) = dNdy;
    B(6,1:3:24) = dNdz; B(6,3:3:24) = dNdx;
    
    ke = ke + B'*D*B*detJ;  % Element stiffness matrix
end
