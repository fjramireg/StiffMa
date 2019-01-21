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

function ke = Hex8vectorSym(X,D,L)
% Symmetric part of the element stiffness matrix ke for a VECTOR problem
B  = zeros(6,24,class(X));  % Initializes the matrix B
ke = zeros(300,1,class(X)); % Initializes the element stiffness matrix
for i=1:8                   % Loop over numerical integration points
    Li = L(:,:,i);          % Matrix L in point i
    Jac  = Li*X;            % Jacobian matrix
    detJ = det(Jac);        % Jacobian's determinant
    dNdX = Jac\Li;          % Shape function derivatives with respect to x,y,z
    % Matrix B
    B(1,1:3:24) = dNdX(1,:);
    B(2,2:3:24) = dNdX(2,:);
    B(3,3:3:24) = dNdX(3,:);
    B(4,1:3:24) = dNdX(2,:);  B(4,2:3:24) = dNdX(1,:);
    B(5,2:3:24) = dNdX(3,:);  B(5,3:3:24) = dNdX(2,:);
    B(6,1:3:24) = dNdX(3,:);  B(6,3:3:24) = dNdX(1,:);
    temp = 0;
    for j=1:24           % Loops to compute the symmetric part of ke
        for k=j:24
            idx = temp + k;
            ke(idx) = ke(idx) + (B(:,k)'*D*B(:,j))*detJ;
        end
        temp = temp + k - j;
    end
end
