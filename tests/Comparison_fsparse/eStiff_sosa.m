function Ke_all = eStiff_sosa(c, nel)
% ESTIFF_SOSA Compute the element stiffness matrix for a SCALAR (s) problem
% by using optimized (o) CPU code for computing the symmety (s) part
% of ke to return ALL (a) elemental matrices.
%
%   KE_ALL = ESTIFF_SOSA(c, nel) returns the element stiffness matrix "ke"
%   for all elements in a finite element analysis of scalar problems in a
%   three-dimensional domain taking advantage of symmetry but with an
%   optimized CPU code, where:
%   - "c" is the material property (thermal consuctivity)
%   - "nel" is the total number of elements in the mesh
%
%   See also STIFFMA_VOCPU
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  September 16, 2020. Version: 1.0
%

L = dNdrst('double');                               % Shape function
X = [-1,-1,-1; 1,-1,-1; 1,1,-1; -1,1,-1; -1,-1,1; 1,-1,1; 1,1,1; -1,1,1];
Ke = eStiff_sss(X,c,L,'double');                    % Local stiffness matrix (symmetric part)
Ke_all = reshape(Ke*[1:nel], length(Ke)*nel, 1);    %#ok All finite elemental matrices
