function [iK, jK] = Index_vsa(elements, sets)
% INDEX_VSA Computes the row/column indices of K for a VECTOR (s) problem using
% SERIAL (s) computing on the CPU to return ALL (a) indices for the mesh.
%   [iK, jK] = INDEX_VSA(elements,dType) returns the rows "iK" and columns
%   "jK" position of all element stiffness matrices in the global system
%   for a finite element analysis of a scalar problem in a three-dimensional
%   domain, where "elements" is the connectivity matrix of size nelx8 and dType
%   is the data type defined to the "elements" array. The struct "sets" must
%   contain several similation parameters:
%   - sets.dTE is the data precision of "Mesh.elements"
%   - sets.nel is the number of finite elements
%   - sets.sz  is the number of symmetry entries
%
%   See also STIFFMA_VS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 07/02/2020. Version: 1.4. Name changed, Doc improved
% 	Modified: 28/01/2019. Version: 1.3
%   Created:  16/01/2019. Version: 1.0

iK = zeros(sets.edof, sets.edof, sets.nel, sets.dTE);           % Stores the rows' indices
jK = zeros(sets.edof, sets.edof, sets.nel, sets.dTE);           % Stores the columns' indices
for e = 1:sets.nel                                             	% Loop over elements
    n = elements(e,:);                                     % Nodes of the element 'e'
    edofs = [3*n-2; 3*n-1; 3*n];                                % DOFs of the element 'e'
    ind = repmat(edofs(:), 1, sets.edof);                       % Index for element 'e'
    iK(:,:,e) = ind;                                            % Row index storage
    jK(:,:,e) = ind';                                           % Columm index storage
end
iK = iK(:);
jK = jK(:);
