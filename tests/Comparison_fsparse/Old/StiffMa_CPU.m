function K = StiffMa_CPU(nelx, nely, nelz, MP)
% STIFFMA_CPU is a function to generate the STIFFnes MAtrix on the CPU for
% the vector problem based on vectorized code developed by Federico Ferrari
% and Ole Sigmund and "fsparse" function developed by Stefan Engblom and
% Dimitar Lukarski.
% 
% K = STIFFMA_CPU(nelx ,nely ,nelz, MP) returns the lower-triangle of a
% sparse matrix K from finite element analysis in vector problems in a
% three-dimensional domain taking advantage of symmetry and optimized CPU 
% code, where the required inputs are:
%   - "nelx" is the number of elements id the X-direction,
%   - "nely" is the number of elements id the Y-direction, 
%   - "nelz" is the number of elements id the Z-direction,
%   - "MP.E" is the Young's modulus and
%   - "MP.nu" is the Poisson ratio are the material property for an isotropic

%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.
%
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  June 21, 2020.    Version: 1.0
%   Modified: June 22, 2020.    Version: 1.1. Material properties are now an input
% 
% Credits:
%   - Assembly procedure:
%   Ferrari, F. & Sigmund, O. (2020), A new generation 99 line Matlab code
%   for compliance topology optimization and its extension to 3D. arXiv.org,
%   https://arxiv.org/abs/2005.05436.
% 
%   - The "fsparse" function is used from "stenglib" library developed by:
%   Engblom, S. & Lukarski, D. (2016). Fast MATLAB compatible sparse
%   assembly on multicore computers. Parallel Computing, 56, 1-17.
%   https://doi.org/10.1016/j.parco.2016.04.001.
%   Code: https://github.com/stefanengblom/stenglib
%

%% Indices computation 
nEl = nelx * nely * nelz ;              % number of elements #3D#
nodeNrs = int32 ( reshape ( 1 : ( 1 + nelx ) * ( 1 + nely ) * ( 1 + nelz ), ...
    1 + nely , 1 + nelz , 1 + nelx ) ); % nodes numbering #3D#
cVec = reshape ( 3 * nodeNrs ( 1 : nely , 1 : nelz , 1 : nelx ) + 1, nEl , 1 ); % #3D#
cMat = cVec + int32 ( [0 ,1 ,2 ,3*( nely +1) *( nelz +1) +[0 ,1 ,2 , -3 , -2 , -1] , -3 , -2 , -1 ,3*( nely +...
    1) +[0 ,1 ,2] ,3*( nely +1) *( nelz +2) +[0 ,1 ,2 , -3 , -2 , -1] ,3*( nely +1) +[ -3 , -2 , -1]]);% connectivity matrix #3D#
nDof = ( 1 + nely ) * ( 1 + nelz ) * ( 1 + nelx ) * 3; % total number of DOFs #3D#
[ sI , sII ] = deal ( [ ] );
for j = 1 : 24
    sI = cat ( 2, sI , j : 24 );
    sII = cat ( 2, sII , repmat ( j, 1, 24 - j + 1 ) );
end
[ iK , jK ] = deal ( cMat ( :, sI )', cMat ( :, sII )' );
Iar = sort ( [ iK( : ), jK( : ) ], 2, 'descend' ); clear iK jK % reduced assembly indexing

%% Numerical integration (only for vector problem in structured meshes)
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
sK = reshape ( Ke( : ) * [1:nEl] , length ( Ke ) * nEl , 1 );       %#ok All finite elemental matrices

%% Assembly
K = fsparse ( Iar( :, 1 ), Iar ( :, 2 ), sK , [ nDof , nDof ] );