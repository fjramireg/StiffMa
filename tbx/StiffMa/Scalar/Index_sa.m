function [iK, jK] = Index_sa(elements, sets)
% INDEX_SA Computes the row/column indices of K for a SCALAR (s) problem on the
% CPU to return ALL (a) indices for the mesh (full matrix). This is a vectorized function.
% 
%   [iK, jK] = INDEX_SA(elements,sets) returns the rows "iK" and columns "jK"
%   position of all element stiffness matrices in the global system for a finite
%   element analysis of a scalar problem in a three-dimensional domain, where
%   "elements" is the connectivity matrix of size nelx8. The struct "sets" must
%   contain several simulation parameters:
%   - sets.nel is the number of finite elements
%   - sets.edof is the number of DOFs per element
%
%   See also INDEX_SPS, INDEX_SSS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Institución Universitaria Pascual Bravo, Medellin-Colombia
%   Universidad Nacional de Colombia - Medellin
%       Modified: July 27, 2026. Improved doc
%       Created:  18/02/2020. Version: 1.0

if ( size(elements,2)~=8 )	% Check if the array "elements" is nel-by-8
    error('Input "elements" must be a nelx8 array');
end

iK = reshape(repmat(elements',sets.edof,1), sets.edof^2*sets.nel, 1);        % Computes & stores the row indices
jK = reshape(repmat(elements(:),1,sets.edof)', sets.edof^2*sets.nel, 1);    % Computes & stores the column indices
