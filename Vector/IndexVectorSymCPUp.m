function [iK, jK] = IndexVectorSymCPUp(elements)
% INDEXVECTORSYMCPUP Compute the row and column indices of lower symmetric
% part of global stiffness matrix for a VECTOR problem taking advantage of
% symmetric and parallel computing on multicore CPU processors.
%   INDEXVECTORSYMCPUP(elements) returns the rows "iK" and columns "jK"
%   position of all element stiffness matrices in the global system for a
%   finite element analysis of a vector problem in a three-dimensional
%   domain taking advantage of symmetry and parallel computing on multicore
%   CPU processors,, where "elements" is the connectivity matrix.
%
%   See also STIFFMATGENVCSYMCPUP, INDEXVECTORSYMCPU, INDEXVECTORSYMGPU
%
%   For more information, see <a href="matlab:
%   web('https://github.com/fjramireg/MatGen')">the MatGen Web site</a>.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created: 16/01/2019. Modified: 28/01/2019. Version: 1.3

dType = class(elements);            % Data type
nel = size(elements,1);            	% Number of elements
iK  = zeros(300,nel,dType);         % Stores row indices
jK  = zeros(300,nel,dType);         % Stores column indices
parfor e = 1:nel
    [iK(:,e), jK(:,e)] = dofsmapping(elements(e,:),dType);
end
iK = iK(:);
jK = jK(:);

function [rows, cols] = dofsmapping(n,dType)
dof = [3*n-2; 3*n-1; 3*n];
rows = zeros(300,1,dType);
cols = zeros(300,1,dType);
temp = 0;
for j = 1:24
    for i = j:24
        idx = temp + i;
        if dof(i) >= dof(j)
            rows(idx) = dof(i);
            cols(idx) = dof(j);
        else
            rows(idx) = dof(j);
            cols(idx) = dof(i);
        end
    end
    temp = temp + i - j;
end
