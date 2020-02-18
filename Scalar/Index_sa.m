function [iK, jK] = Index_sa(elements, sets)
% INDEX_SA Computes the row/column indices of K for a SCALAR (s) problem on the
% CPU to return ALL (a) indices for the mesh. This is a vectorized function.
%   [iK, jK] = INDEX_SA(elements,sets) returns the rows "iK" and columns "jK"
%   position of all element stiffness matrices in the global system for a finite
%   element analysis of a scalar problem in a three-dimensional domain, where
%   "elements" is the connectivity matrix of size 8xnel. The struct "sets" must
%   contain several similation parameters:
%   - sets.nel is the number of finite elements
%   - sets.edof is the number of DOFs per element
%
%   See also INDEX_SPS, INDEX_SSS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  18/02/2020. Version: 1.0

if ( size(elements,1)~=8 )	% Check if the array "elements" is 8xnel
    error('Input "elements" must be a 8xnel array');
end

iK = reshape(repmat(elements,sets.edof,1), sets.edof^2*sets.nel, 1);        % Computes & stores the row indices
jK = reshape(repmat(elements(:),1,sets.edof)', sets.edof^2*sets.nel, 1);    % Computes & stores the column indices
