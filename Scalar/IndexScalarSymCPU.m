function [iK, jK] = IndexScalarSymCPU(elements)
% INDEXSCALARSYMCPU Compute the row and column indices of lower symmetric
% part of global stiffness matrix for a SCALAR problem.
%   INDEXSCALARSYMCPU(elements) returns the rows "iK" and columns "jK" position
%   of all element stiffness matrices in the global system for a finite
%   element analysis of a scalar problem in a three-dimensional domain
%   taking advantage of symmetry, where "elements" is the connectivity
%   matrix.
%
%   See also STIFFMATGENSCSYM, STIFFMATGENSCSYMGPU
%
%   For more information, see <a href="matlab:
%   web('https://github.com/fjramireg/MatGen')">the MatGen Web site</a>.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created: 30/11/2018. Modified: 21/01/2019. Version: 1.3

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
