%  * ====================================================================*/
% ** This function was developed by:
%  *          Francisco Javier Ramirez-Gil
%  *          Universidad Nacional de Colombia - Medellin
%  *          Department of Mechanical Engineering
%  *
%  ** Please cite this code as:
%  *
%  ** Date & version
%  *      Created: 30/11/2018. Last modified: 21/01/2019
%  *      V 1.3
%  *
%  * ====================================================================*/

function ke = Hex8scalar(X,c)
% Element stiffness matrix ke for a SCALAR problem
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
