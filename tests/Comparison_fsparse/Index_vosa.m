function [Iar, nel, tdof] = Index_vosa(nelx, nely, nelz)
% INDEX_VOSA Compute the row/column indices of tril(K) in a vector (v)
% problem using optimized (o) CPU code for computing the symmety (s) part
% of K to return ALL (a) indices for the mesh.
%   [Iar, nel, tdof] = INDEX_VOSA(nelx, nely, nelz) returns the rows "iK"
%   and columns "jK" position in a matrix called Iar (i.e. Iar=[iK,jK]) of
%   all element stiffness matrices in the global system for a finite
%   element analysis of a vector problem in a three-dimensional 
%   domain taking advantage of symmetry, where:
%   - "nelx" is the number of elements id the X-direction,
%   - "nely" is the number of elements id the Y-direction,
%   - "nelz" is the number of elements id the Z-direction,
%   - "nel"  is the total number of elements
%   - "tdof" is the total number of DOFs
% The mesh is sopposed to be structured with 8-noded hexaedral elements. 
%
%   See also STIFFMA_CPU
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Modified: August 20, 2020. Version: 1.1. Impoved doc and readability
%   Created:  June 22, 2020. Version: 1.0
%
% Credits:
%   Ferrari, F. & Sigmund, O. (2020), A new generation 99 line Matlab code
%   for compliance topology optimization and its extension to 3D. arXiv.org,
%   https://arxiv.org/abs/2005.05436.
%

% Global variable
nel = nelx * nely * nelz;	% number of elements
nx = (1 + nelx);            % number of nodes in the X-dir
ny = (1 + nely);            % number of nodes in the Y-dir
nz = (1 + nelz);            % number of nodes in the Z-dir
nn = (ny * nz * nx);        % total number of nodes in the mesh
nnxe = 8;                   % number of nodes per element (8-noded hexahedral element)
ndof = 3;                   % number of DOFs per node (3 DOF per node = vector problem)
tdof = ndof*nn;             % total number of DOFs
emsz = ndof*nnxe;           % element matrix size

% Index computation
nodeNrs = int32(reshape(1:nn, ny, nz, nx));                         % nodes numbering in 3D
cVec = reshape(ndof*nodeNrs(1:nely, 1:nelz, 1:nelx) + 1, nel, 1);   % 3D#
cMat = cVec + int32([0, 1, 2, 3*ny*nz+[0, 1, 2, -3, -2, -1], -3, -2, -1,...
    3*ny+[0, 1, 2], 3*ny*(nelz+2)+[0, 1, 2, -3, -2, -1], 3*ny+[-3, -2, -1]]);% connectivity matrix #3D#
[ sI , sII ] = deal ( [ ] );
for j = 1 : 24
    sI = cat ( 2, sI , j : 24 );
    sII = cat ( 2, sII , repmat ( j, 1, 24 - j + 1 ) );
end
[ iK , jK ] = deal ( cMat ( :, sI )', cMat ( :, sII )' );
Iar = sort ( [ iK( : ), jK( : ) ], 2, 'descend' );
