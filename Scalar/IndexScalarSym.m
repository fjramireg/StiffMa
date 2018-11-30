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

function [iK, jK] = IndexScalarSym(elements)
% Row/column indices of tril(K) (SCALAR)
elements = uint32(elements);
nel = size(elements,1);
iK  = zeros(36*nel,1,'uint32');
jK  = zeros(36*nel,1,'uint32');
for e = 1:nel
    n = elements(e,:);
    temp = 0;
    for j = 1:8
        for i = j:8
            idx = temp + i + 36*(e-1);
            if n(i) > n(j)
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
