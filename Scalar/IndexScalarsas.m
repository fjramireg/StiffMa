function [iK, jK] = IndexScalarsas(elements, settings)
% INDEXSCALARSAS Compute the row/column indices of tril(K) using SERIAL computing
% for a SCALAR problem on the CPU.
%   [iK, jK] = INDEXSCALARSAS(elements,dType) returns the rows "iK" and columns "jK"
%   position of all element stiffness matrices in the global system for a finite
%   element analysis of a scalar problem in a three-dimensional domain taking
%   advantage of symmetry, where "elements" is the connectivity matrix of size
%   nelx8 and dType is the data type defined to the "elements" array. 
%
%   See also STIFFMASS, INDEXSCALARSAP
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 05/12/2019. Version: 1.4. Name changed, Doc improved
% 	Modified: 21/01/2019. Version: 1.3
%   Created:  30/11/2018. Version: 1.0

iK  = zeros(36*settings.nel, 1, settings.dTE);  % Row indices
jK  = zeros(36*settings.nel, 1, settings.dTE);  % Column indices
for e = 1:settings.nel
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
