function [iK, jK] = Index_ssat(elements, sets)
% INDEX_SSAT Computes the row/column indices of K for a SCALAR (s) problem using
% SERIAL (s) computing on the CPU to return ALL (a) indices for the mesh to form
% only the lower triangular (T) part of K (tril(K)). % This vectorized version
% matches the layout of the CUDA form version 2 (Index_spsa_opt). That is the
% coalesced-style layout: idx = t*nel + e, where iK and jK are of size sets.nel x sets.sz
%
%   [iK, jK] = INDEX_SSAT(elements,dType) returns the rows "iK" and columns "jK"
%   position of all element stiffness matrices in the global system for a finite
%   element analysis of a scalar problem in a three-dimensional domain, where
%   "elements" is the connectivity matrix of size nelx8. The struct "sets" must
%   contain several simulation parameters:
%   - sets.dTE is the data precision of "Mesh.elements"
%   - sets.nel is the number of finite elements
%   - sets.edof is the number of DOFs per element
%
%   See also STIFFMA_SSS, INDEX_SPS, INDEX_SSS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Institución Universitaria Pascual Bravo, Medellin-Colombia
%   Created:  April 20, 2026. Version: 1.0

% Indices of the lower triangular part in column-wise order
[i, j] = find(tril(true(sets.edof)));  % size: 1-by-sz

% Vectorized extraction for all elements at once
A = elements(:, i); % size: nel-by-sz
B = elements(:, j); % size: nel-by-sz

iK = reshape(max(A,B).', [], 1); % size: nel x sz, 1
jK = reshape(min(A,B).', [], 1); % size: nel x sz, 1
