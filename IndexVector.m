function [iK, jK] = IndexVector(elements)
% Row/column indices of the lower triangular sparse matrix K (VECTOR)
elements = uint32(elements);        % Converts the precision data
nel = size(elements,1);             % Number of FEs
iK  = zeros(300*nel,1,'uint32');    % Store row indices
jK  = zeros(300*nel,1,'uint32');    % Store column indices
for e = 1:nel
    n = sort(elements(e,:));
    dof = [3*n-2; 3*n-1; 3*n];
    temp = 0;
    for j = 1:24
        for i = j:24
            idx = temp + i + 300*(e-1);
            iK(idx) = dof(i);
            jK(idx) = dof(j);
        end
        temp = temp + i - j;
    end
end
