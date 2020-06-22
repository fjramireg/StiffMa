function sK = eStiff_vosa(MP, nel)
% ESTIFF_VOSA Compute the element stiffness matrix for a VECTOR (v) problem
% by using optimized (o) CPU code for computing the symmety (s) part
% of ke to return ALL (a) elemental matrices.
%
%   ESTIFF_VOSA(MP, nel) returns the element stiffness matrix "ke" for all
%   elements in a finite element analysis of vector problems in a
%   three-dimensional domain taking advantage of symmetry but with an
%   optimized CPU code, where:
%   - "MP.E" is the Young's modulus and
%   - "MP.nu" is the Poisson ratio are the material property for an isotropic
%
%   See also STIFFMA_CPU
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  June 22, 2020. Version: 1.0
%
% Credits:
%   Ferrari, F. & Sigmund, O. (2020), A new generation 99 line Matlab code
%   for compliance topology optimization and its extension to 3D. arXiv.org,
%   https://arxiv.org/abs/2005.05436.
%

% Numerical integration (only for vector problem in structured meshes)
Ke = MP.E/(1+ MP.nu) /(2* MP.nu -1) /144 *( [ -32; -6; -6;8;6;6;10;6;3; -4; -6; -3; -4; -3; -6;10;...
    3;6;8;3;3;4; -3; -3; -32; -6; -6; -4; -3;6;10;3;6;8;6; -3; -4; -6; -3;4; -3;3;8;3;...
    3;10;6; -32; -6; -3; -4; -3; -3;4; -3; -6; -4;6;6;8;6;3;10;3;3;8;3;6;10; -32;6;6;...
    -4;6;3;10; -6; -3;10; -3; -6; -4;3;6;4;3;3;8; -3; -3; -32; -6; -6;8;6; -6;10;3;3;4;...
    -3;3; -4; -6; -3;10;6; -3;8;3; -32;3; -6; -4;3; -3;4; -6;3;10; -6;6;8; -3;6;10; -3;...
    3;8; -32; -6;6;8;6; -6;8;3; -3;4; -3;3; -4; -3;6;10;3; -6; -32;6; -6; -4;3;3;8; -3;...
    3;10; -6; -3; -4;6; -3;4;3; -32;6;3; -4; -3; -3;8; -3; -6;10; -6; -6;8; -6; -3;10; -32;...
    6; -6;4;3; -3;8; -3;3;10; -3;6; -4;3; -6; -32;6; -3;10; -6; -3;8; -3;3;4;3;3; -4;6;...
    -32;3; -6;10;3; -3;8;6; -3;10;6; -6;8; -32; -6;6;8;6; -6;10;6; -3; -4; -6;3; -32;6;...
    -6; -4;3;6;10; -3;6;8; -6; -32;6;3; -4;3;3;4;3;6; -4; -32;6; -6; -4;6; -3;10; -6;3;...
    -32;6; -6;8; -6; -6;10; -3; -32; -3;6; -4; -3;3;4; -32; -6; -6;8;6;6; -32; -6; -6; -4;...
    -3; -32; -6; -3; -4; -32;6;6; -32; -6; -32]+ MP.nu *[ 48;0;0;0; -24; -24; -12;0; -12;0;...
    24;0;0;0;24; -12; -12;0; -12;0;0; -12;12;12;48;0;24;0;0;0; -12; -12; -24;0; -24;...
    0;0;24;12; -12;12;0; -12;0; -12; -12;0;48;24;0;0;12;12; -12;0;24;0; -24; -24;0;...
    0; -12; -12;0;0; -12; -12;0; -12;48;0;0;0; -24;0; -12;0;12; -12;12;0;0;0; -24;...
    -12; -12; -12; -12;0;0;48;0;24;0; -24;0; -12; -12; -12; -12;12;0;0;24;12; -12;0;...
    0; -12;0;48;0;24;0; -12;12; -12;0; -12; -12;24; -24;0;12;0; -12;0;0; -12;48;0;0;...
    0; -24;24; -12;0;0; -12;12; -12;0;0; -24; -12; -12;0;48;0;24;0;0;0; -12;0; -12;...
    -12;0;0;0; -24;12; -12; -12;48; -24;0;0;0;0; -12;12;0; -12;24;24;0;0;12; -12;...
    48;0;0; -12; -12;12; -12;0;0; -12;12;0;0;0;24;48;0;12; -12;0;0; -12;0; -12; -12;...
    -12;0;0; -24;48; -12;0; -12;0;0; -12;0;12; -12; -24;24;0;48;0;0;0; -24;24; -12;...
    0;12;0;24;0;48;0;24;0;0;0; -12;12; -24;0;24;48; -24;0;0; -12; -12; -12;0; -24;...
    0;48;0;0;0; -24;0; -12;0; -12;48;0;24;0;24;0; -12;12;48;0; -24;0;12; -12; -12;...
    48;0;0;0; -24; -24;48;0;24;0;0;48;24;0;0;48;0;0;48;0;48 ] );    % elemental stiffness matrix #3D#
sK = reshape ( Ke( : ) * [1:nel] , length ( Ke ) * nel , 1 );       %#ok All finite elemental matrices
