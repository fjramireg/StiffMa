function ke = Hex8vectorSym(X,D,L)
% Symmetric part of the element stiffness matrix ke for a VECTOR problem
B  = zeros(6,24);       % Initializes the matrix B
ke = zeros(300,1);      % Initializes the element stiffness matrix
for i=1:8               % Loop over numerical integration points
    Li = L(:,:,i);      % Matrix L in point i
    Jac  = Li*X;        % Jacobian matrix
    detJ = det(Jac);    % Jacobian's determinant
    dNdX = Jac\Li;      % Shape function derivatives with respect to x,y,z
    % Matrix B
    B(1,1:3:24) = dNdX(1); B(2,2:3:24) = dNdX(2); B(3,3:3:24) = dNdX(3);
    B(4,1:3:24) = dNdX(2);  B(4,2:3:24) = dNdX(1);
    B(5,2:3:24) = dNdX(3);  B(5,3:3:24) = dNdX(2);
    B(6,1:3:24) = dNdX(3);  B(6,3:3:24) = dNdX(1);
    temp = 0;
    for j=1:24           % Loops to compute the symmetric part of ke
        for k=j:24
            idx = temp + k;
            ke(idx) = ke(idx) + (B(:,j)'*D*B(:,k))*detJ;
        end
        temp = temp + k - j;
    end
end
