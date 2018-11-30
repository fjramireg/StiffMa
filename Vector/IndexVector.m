function [iK, jK] = IndexVector(elements)
% Row/column indices of the lower triangular sparse matrix K (VECTOR)
elements = uint32(elements);        % Converts the precision data
nel = size(elements,1);             % Number of elements
iK  = zeros(300*nel,1,'uint32');    % Stores row indices
jK  = zeros(300*nel,1,'uint32');    % Stores column indices
for e = 1:nel
    n = elements(e,:);
    dof = [3*n-2; 3*n-1; 3*n];
    temp = 0;
    for j = 1:24
        for i = j:24
            idx = temp + i + 300*(e-1);
            if dof(i) > dof(j)
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
