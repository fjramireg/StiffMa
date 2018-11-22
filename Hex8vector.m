function ke = Hex8vector(X,D)
% Element stiffnes matrix ke for a VECTOR problem
p = 1/sqrt(3);              % Gauss points
r = [p,-p,p,-p,p,-p,p,-p];  % Points through r-coordinate
s = [p,p,-p,-p,p,p,-p,-p];  % Points through s-coordinate
t = [p,p,p,p,-p,-p,-p,-p];  % Points through t-coordinate
ke = zeros(24,24);          % Initialize the element stiffness matrix
B  = zeros(6,24);           % Initialize the matrix B
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
    dNdX = Jac\L;           % Shape function derivatives with respect to x,y,z
    % Matrix B
    B(1,1:3:24) = dNdX(1);
    B(2,2:3:24) = dNdX(2);
    B(3,3:3:24) = dNdX(3);
    B(4,1:3:24) = dNdX(2);  B(4,2:3:24) = dNdX(1);
    B(5,2:3:24) = dNdX(3);  B(5,3:3:24) = dNdX(2);
    B(6,1:3:24) = dNdX(3);  B(6,3:3:24) = dNdX(1);    
    ke = ke + B'*D*B*detJ;  % Element stiffness matrix    
end
