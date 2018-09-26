function ke = Hex8scalarSym(X,c,L)
% Element stiffnes symmetric matrix ke for a SCALAR problem

% Numerical integrationspa
ke = zeros(36,1);
for i=1:8
    Li = L(:,:,i);
    % Jacobian, its determinant and its inverse
    Jac  = Li*X;
    detJ = det(Jac);
    %  Shape function derivates with respect to x,y,z
    B = Jac\Li;
    % Element stiffness matrix
    temp = 0;
    for j=1:8
        for k=j:8
            idx = temp+k;
            ke(idx) = ke(idx) + c*detJ*(B(:,j)'*B(:,k));
        end
        temp = temp + k-j;
    end
end
