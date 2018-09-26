function [iK, jK] = IndexScalar(elements)
% Computation of the indices of the sparse matrix
elements = uint32(elements);    % Converts the precision data
nel = size(elements,1);         % Number of FEs
iK  = zeros(36*nel,1,'uint32'); % Store row indices
jK  = zeros(36*nel,1,'uint32'); % Store column indices
for e = 1:nel
    n = elements(e,:);
    temp = 0;
    for j = 1:8
        for i = j:8
            idx = temp+i + 36*(e-1);
            iK(idx) = n(i);
            jK(idx) = n(j);
        end
        temp = temp + i-j;
    end
end
