function [iK, jK] = IndexScalar(elements)
% Row/column indices of the lower triangular sparse matrix K (SCALAR)
elements = uint32(elements);        % Converts the precision data
nel = size(elements,1);             % Number of elements
iK  = zeros(36*nel,1,'uint32');     % Stores the row indices
jK  = zeros(36*nel,1,'uint32');     % Stores the column indices
for e = 1:nel
    n = sort(elements(e,:));
    temp = 0;
    for j = 1:8
        for i = j:8
            idx = temp + i + 36*(e-1);
            iK(idx) = n(i);
            jK(idx) = n(j);
        end
        temp = temp + i - j;
    end
end
