function spy3D(S,width)
%SPY3D Visualize sparsity pattern with 3D bars.
%   SPY3D(S) plots the sparsity pattern of the matrix S in a three-dimensional
%   (3D) domain with matrix entries represented as bars that are colored
%   according to their value (height) and using a colorbar.
%
%   SPY3D(S,width) sets the width of the bars and controls the separation of
%   bars within a group. The default width is 0.8 and the bars have a slight
%   separation. If width is 1, the bars within a group touch one another.
%
%   See also SPY, BAR3, COLORBAR.
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Modified: 17/12/2019. Version: 1.4
%   Created: 30/11/2018.

if nargin == 1; width = 0.8; end
b = bar3(S,width);
for k = 1:length(b)
    b(k).CData = b(k).ZData;
    b(k).FaceColor = 'interp';
    b(k).EdgeAlpha=0.0;
    b(k).FaceAlpha=0.5;
end
colorbar;
title(['nz = ' int2str(nnz(S))]);
