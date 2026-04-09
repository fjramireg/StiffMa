function Ke = eStiff_sosa2(c)
% ESTIFF_SOSA2 Compute the element stiffness matrix for a SCALAR (s) problem
% by using optimized (o) CPU code for computing the symmety (s) part of ke.
%
%   KE = ESTIFF_SOSA2(c) returns the element stiffness matrix "ke" for one
%   Hex8 element in a finite element analysis of scalar problems in a
%   three-dimensional domain taking advantage of symmetry but with an 
%   optimized CPU code, where:
%   - "c" is the material property (thermal consuctivity)
%   - "nel" is the total number of elements in the mesh
%
%   See also STIFFMA_SOCPU2
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  September 17, 2020. Version: 1.0
%

L = dNdrst('double');                               % Shape function
X = [-1,-1,-1; 1,-1,-1; 1,1,-1; -1,1,-1; -1,-1,1; 1,-1,1; 1,1,1; -1,1,1];
Ke = eStiff_sss(X,c,L,'double');                    % Local stiffness matrix (symmetric part)
