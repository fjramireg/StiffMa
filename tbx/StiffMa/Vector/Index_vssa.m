function [iK, jK] = Index_vssa(elements, sets)
% INDEX_VSSA Compute the row/column indices of tril(K) in a vector (v) problem
% using SERIAL (s) computing and symmety (s) to return ALL (a) indices for the mesh.
%   INDEX_VSSA(elements, sets) returns the rows "iK" and columns "jK"
%   position of all element stiffness matrices in the global system for a
%   finite element analysis of a vector problem in a three-dimensional
%   domain taking advantage of symmetry, where "elements" is the
%   connectivity matrix. The struct "sets" must contain several similation
%   parameters:
%   - sets.dTE is the data precision of "Mesh.elements"
%   - sets.nel is the number of finite elements
%   - sets.edof is the number of DOFs per element
%   - sets.sz  is the umber of symmetry entries.
%
%   See also STIFFMA_VSS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 28/01/2020. Version: 1.4. Name changed, Doc improved
% 	Modified: 28/01/2019. Version: 1.3
%   Created:  16/01/2019. Version: 1.0

iK = zeros(sets.sz*sets.nel, 1, sets.dTE);   % Stores row indices
jK = zeros(sets.sz*sets.nel, 1, sets.dTE);   % Stores column indices
for e = 1:sets.nel
    n = elements(e,:);
    dof = [3*n-2; 3*n-1; 3*n];
    temp = 0;
    for j = 1:sets.edof
        for i = j:sets.edof
            idx = temp + i + sets.sz*(e-1);
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
