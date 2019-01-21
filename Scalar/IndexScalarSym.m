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

function [iK, jK] = IndexScalarSym(elements)
% Row/column indices of tril(K) (SCALAR)
dType = class(elements);        % Data type
nel = size(elements,1);         % # of elements
iK  = zeros(36*nel, 1, dType);  % Row indices
jK  = zeros(36*nel, 1, dType);  % Column indices
for e = 1:nel
    n = elements(e,:);
    temp = 0;
    for j = 1:8
        for i = j:8
            idx = temp + i + 36*(e-1);
            if n(i) >= n(j)
                iK(idx) = n(i);
                jK(idx) = n(j);
            else
                iK(idx) = n(j);
                jK(idx) = n(i);
            end
        end
        temp = temp + i - j;
    end
end
