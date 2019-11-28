function [iK, jK] = IndexScalarSymCPUp(elements)
% INDEXSCALARSYMCPUP Compute the row and column indices of lower symmetric
% part of global stiffness matrix for a SCALAR problem taking advantage of
% symmetric and parallel computing on multicore CPU processors.
%   INDEXSCALARSYMCPUP(elements) returns the rows "iK" and columns "jK"
%   position of all element stiffness matrices in the global system for a
%   finite element analysis of a scalar problem in a three-dimensional
%   domain taking advantage of symmetry and parallel computing on multicore CPU
%   processors, where "elements" is the connectivity matrix.
%
%   See also STIFFMATGENSCSYMCPUP, INDEXSCALARSYMCPU, INDEXSCALARSYMGPU
%
%   For more information, see <a href="matlab:
%   web('https://github.com/fjramireg/MatGen')">the MatGen Web site</a>.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created: 28/01/2019. Modified: 28/01/2019. Version: 1.3

dType = class(elements);        % Data type
nel = size(elements,1);         % # of elements
iK  = zeros(36,nel,dType);      % Row indices
jK  = zeros(36,nel,dType);      % Column indices
parfor e = 1:nel
    [iK(:,e), jK(:,e)] = dofsmapping(elements(e,:),dType);
end
iK = iK(:);
jK = jK(:);

function [rows, cols] = dofsmapping(dofs,dType)
rows = zeros(36,1,dType);
cols = zeros(36,1,dType);
temp = 0;
for j = 1:8
    for i = j:8
        idx = temp + i;
        if dofs(i) >= dofs(j)
            rows(idx) = dofs(i);
            cols(idx) = dofs(j);
        else
            rows(idx) = dofs(j);
            cols(idx) = dofs(i);
        end
    end
    temp = temp + i - j;
end