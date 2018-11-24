function [iK, jK] = IndexScalar(elements)
% Row/column indices of the lower triangular sparse matrix K (SCALAR)
elements = uint32(elements);        % Converts the precision data
nel = size(elements,1);             % Number of elements
iK  = zeros(36*nel,1,'uint32');     % Stores row indices
jK  = zeros(36*nel,1,'uint32');     % Stores column indices
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
