%  * ====================================================================*/
% ** This function was developed by:
%  *          Francisco Javier Ramirez-Gil
%  *          Universidad Nacional de Colombia - Medellin
%  *          Department of Mechanical Engineering
%  *
%  ** Please cite this code as:
%  *
%  ** Date & version
%  *      30/11/2018.
%  *      V 1.2
%  *
%  * ====================================================================*/

function ke = Hex8scalarSym(X,c,L)
% Symmetric part of the element stiffness matrix ke for a SCALAR problem
ke = zeros(36,1,class(X));% Initializes the element stiffness matrix
for i=1:8               % Loop over numerical integration
    Li = L(:,:,i);      % Matrix L in point i
    Jac  = Li*X;        % Jacobian matrix
    detJ = det(Jac);    % Jacobian's determinant
    B = Jac\Li;         % B matrix
    temp = 0;
    for j=1:8           % Loops to compute the symmetric part of ke
        for k=j:8
            idx = temp + k;
            ke(idx) = ke(idx) + c*detJ*(B(:,k)'*B(:,j));
        end
        temp = temp + k - j;
    end
end
