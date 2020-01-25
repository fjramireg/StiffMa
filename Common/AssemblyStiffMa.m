function K = AssemblyStiffMa(iK, jK, Ke, dTE, dTN)
% ASSEMBLYSTIFFMA Assembly a global sparse stiffness matrix K.
%   ASSEMBLYSTIFFMA(iK,jK,Ke,dTE,dTN) returns a sparse matrix K that is computed
%   on the CPU or on the GPU according to the input data, where "iK", "jK", and
%   "Ke" are column vectors containing the row index,  colomn index and non-zero
%   value of each entry of the sparse matrix. Whilst dTE and dTN are the data
%   type defined to connectivity and nodal coordinates matrices, respectively. 
%
%   See also SPARSE, ACCUMARRAY
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 07/12/2019. Version: 1.4. Doc improved
% 	Modified: 21/01/2019. Version: 1.3
%   Created:  10/12/2018. Version: 1.0

%% Assembly of global sparse matrix
if ( strcmp(dTE,'double') && strcmp(dTN,'double') )
    K = sparse(iK, jK, Ke);
    
elseif ( (strcmp(dTE,'uint32') || strcmp(dTE,'uint64')) && strcmp(dTN,'double') )
    K = accumarray([iK,jK], Ke, [], [], [], 1);
    
else
    error('MATLAB currently does not support "single" data precision for sparse matrices!');
end
