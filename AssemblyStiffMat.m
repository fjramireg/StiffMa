function K = AssemblyStiffMat(iK,jK,Ke,N,dTE,dTN)
% ASSEMBLYSTIFFMAT Assembly a global sparse stiffness matrix K.
%   ASSEMBLYSTIFFMAT(iK,jK,Ke,N,dTE,dTN) returns a sparse matrix K that is
%   computed on the CPU or on the GPU according to the input data, where
%   "iK", "jK", and "Ke" are column vectors containing the row index,
%   colomn index and non-zero value of each entry of the sparse matrix.
%   Whilst "N" is the size of the matrix, and dTE/dTN are the data type
%   defined to connectivity and nodal coordinates matrices, respectively. 
%
%   See also SPARSE, ACCUMARRAY, ASSEMBLYSCALAR, ASSEMBLYSCALARSYMGPU
%
%   For more information, see <a href="matlab:
%   web('https://github.com/fjramireg/MatGen')">the MatGen Web site</a>.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 05/12/2019. Version: 1.4.
% 	Modified: 21/01/2019. Version: 1.3
%   Created:  10/12/2018. Version: 1.0

%% Assembly of global sparse matrix
if ( strcmp(dTE,'double') && strcmp(dTN,'double') )
    K = sparse(iK, jK, Ke, N, N);
else
    K = accumarray([iK,jK], Ke, [N,N], [], [], 1);
end
