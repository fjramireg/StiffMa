%  * ====================================================================*/
% ** This function was developed by:
%  *          Francisco Javier Ramirez-Gil
%  *          Universidad Nacional de Colombia - Medellin
%  *          Department of Mechanical Engineering
%  *
%  ** Please cite this code as:
%  *
%  ** Date & version
%  *      Created: 17/01/2019. Last modified: 21/01/2019
%  *      V 1.3
%  *
%  * ====================================================================*/

function [iK, jK] = IndexVectorSym(elements)
% Row/column indices of the lower triangular sparse matrix K (VECTOR)
dType = class(elements);         % Data type
nel = size(elements,1);          % Number of elements
iK  = zeros(300*nel,1,dType);    % Stores row indices
jK  = zeros(300*nel,1,dType);    % Stores column indices
for e = 1:nel
    n = elements(e,:);
    dof = [3*n-2; 3*n-1; 3*n];
    temp = 0;
    for j = 1:24
        for i = j:24
            idx = temp + i + 300*(e-1);
            if dof(i) >= dof(j)
                iK(idx) = dof(i);
                jK(idx) = dof(j);
            else
                iK(idx) = dof(j);
                jK(idx) = dof(i);
            end
        end
        temp = temp + i - j;
    end
end
