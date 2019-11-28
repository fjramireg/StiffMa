function [iK, jK] = IndexVectorSymCPU(elements)
% INDEXVECTORSYMCPU Compute the row and column indices of lower symmetric
% part of global stiffness matrix for a VECTOR problem.
%   INDEXVECTORSYMCPU(elements) returns the rows "iK" and columns "jK"
%   position of all element stiffness matrices in the global system for a
%   finite element analysis of a vector problem in a three-dimensional
%   domain taking advantage of symmetry, where "elements" is the connectivity
%   matrix.
%
%   See also STIFFMATGENVCSYMCPU, STIFFMATGENVCSYMGPU
%
%   For more information, see <a href="matlab:
%   web('https://github.com/fjramireg/MatGen')">the MatGen Web site</a>.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created: 16/01/2019. Modified: 28/01/2019. Version: 1.3

dType = class(elements);         % Data type
nel = size(elements,1);          % Number of elements
iK  = zeros(300*nel,1,dType);    % Stores row indices
jK  = zeros(300*nel,1,dType);    % Stores column indices
for e = 1:nel
    n = elements(e,:);
    dof = [3*n-2; 3*n-1; 3*n];
    temp = 0;
    for j = 1:24
        for i = j:24
            idx = temp + i + 300*(e-1);
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
