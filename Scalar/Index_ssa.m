function [iK, jK] = Index_ssa(elements, sets)
% INDEX_SSA Computes the row/column indices of K for a SCALAR (s) problem using
% SERIAL (s) computing on the CPU to return ALL (a) indices for the mesh.
%   [iK, jK] = INDEX_SSA(elements,dType) returns the rows "iK" and columns "jK"
%   position of all element stiffness matrices in the global system for a finite
%   element analysis of a scalar problem in a three-dimensional domain, where
%   "elements" is the connectivity matrix of size nelx8. The struct "sets" must
%   contain several similation parameters: 
%   - sets.dTE is the data precision of "Mesh.elements"
%   - sets.nel is the number of finite elements
%   - sets.edof is the number of DOFs per element
%
%   See also STIFFMA_SSS, INDEX_SPS, INDEX_SSS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 07/02/2020. Version: 1.4. Name changed, Doc improved
% 	Modified: 21/01/2019. Version: 1.3
%   Created:  30/11/2018. Version: 1.0

iK = zeros(sets.edof, sets.edof, sets.nel, sets.dTE);       % Stores the rows' indices
jK = zeros(sets.edof, sets.edof, sets.nel, sets.dTE);       % Stores the columns' indices
for e = 1:sets.nel                                          % Loop over elements
    ind = repmat(elements(e,:),sets.edof,1);                % Index for element 'e'
    iK(:,:,e) = ind';                                       % Row index storage
    jK(:,:,e) = ind;                                        % Columm index storage
end
iK = iK(:);
jK = jK(:);
