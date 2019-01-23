function spy3D(S)
%SPY3D Visualize sparsity pattern with 3D bars.
%   SPY3D(S) plots the sparsity pattern of the matrix S in a
%   three-dimensional (3D) domain with matrix entries represented as bars
%   that are colored according to their value (height) and using a
%   colorbar.
%
%   See also SPY, BAR3 COLORBAR.
%
%   For more information, see <a href="matlab:
%   web('https://github.com/fjramireg/MatGen')">the MatGen Web site</a>.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created: 30/11/2018. Modified: 21/01/2019. Version: 1.3

b = bar3(S,1);
colorbar;
for k = 1:length(b)
    zdata = b(k).ZData;
    b(k).CData = zdata;
    b(k).FaceColor = 'interp';
end
title(['nz = ' int2str(nnz(S))]);
