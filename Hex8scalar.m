function ke = Hex8scalar(X,c)
% Element stiffnes matrix ke for a SCALAR problem

% Gauss points table
p = 1/sqrt(3);
r = [p,-p,p,-p,p,-p,p,-p];
s = [p,p,-p,-p,p,p,-p,-p];
t = [p,p,p,p,-p,-p,-p,-p];

% Numerical integration
ke = zeros(8,8);
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
    
    %  Shape function derivates with respect to x,y,z
    B = Jac\L;
    
    % Element stiffness matrix
    ke = ke + c*detJ*(B'*B);
end
