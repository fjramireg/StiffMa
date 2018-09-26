function ke = Hex8vector(X,D)
% Stiffnes matrix for a VECTOR problem

% Gauss points table
p = 1/sqrt(3);
r = [p,-p,p,-p,p,-p,p,-p];
s = [p,p,-p,-p,p,p,-p,-p];
t = [p,p,p,p,-p,-p,-p,-p];

% Numerical integration
ke = zeros(24,24);  B  = zeros(6,24);
for i=1:8
    
    ri = r(i); si = s(i); ti = t(i);
    
    %  Shape function derivates with respect to r,s,t
    dNdr = (1/8)*[-(1-si)*(1-ti),  (1-si)*(1-ti), (1+si)*(1-ti), -(1+si)*(1-ti),...
        -(1-si)*(1+ti),  (1-si)*(1+ti), (1+si)*(1+ti), -(1+si)*(1+ti)];
    dNds = (1/8)*[-(1-ri)*(1-ti), -(1+ri)*(1-ti), (1+ri)*(1-ti),  (1-ri)*(1-ti),...
        -(1-ri)*(1+ti), -(1+ri)*(1+ti), (1+ri)*(1+ti),  (1-ri)*(1+ti)];
    dNdt = (1/8)*[-(1-ri)*(1-si), -(1+ri)*(1-si),-(1+ri)*(1+si), -(1-ri)*(1+si),...
        (1-ri)*(1-si),  (1+ri)*(1-si), (1+ri)*(1+si),  (1-ri)*(1+si)];
    
    L = [dNdr; dNds; dNdt];
    
    % Jacobian, its determinant and its inverse
    Jac  = L*X;
    detJ = det(Jac);
    Jinv = inv(Jac);
    
    %  Shape function derivates with respect to x,y,z
    dNdX = Jinv*L;
    
    % Matrix B
    B(1,1:3:24) = dNdX(1);
    B(2,2:3:24) = dNdX(2);
    B(3,3:3:24) = dNdX(3);
    B(4,1:3:24) = dNdX(2);  B(4,2:3:24) = dNdX(1);
    B(5,2:3:24) = dNdX(3);  B(5,3:3:24) = dNdX(2);
    B(6,1:3:24) = dNdX(3);  B(6,3:3:24) = dNdX(1);
    
    % Element stiffness matrix
    ke = ke + B'*D*B*detJ;
end
