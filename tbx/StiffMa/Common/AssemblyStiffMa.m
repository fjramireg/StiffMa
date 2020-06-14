function K = AssemblyStiffMa(iK, jK, Ke, sets)
% ASSEMBLYSTIFFMA Assembly a global sparse stiffness matrix K.
%   ASSEMBLYSTIFFMA(iK,jK,Ke,dTE,dTN) returns a sparse matrix K that is computed
%   on the CPU or on the GPU according to the input data, where "iK", "jK", and
%   "Ke" are column vectors containing the row index,  colomn index and non-zero
%   value of each entry of the sparse matrix. Whilst sets.dTE and sets.dTN
%   are the data type defined to connectivity and nodal coordinates
%   matrices, respectively. sets.tdofs is the total number of degree of
%   freedoms, which dtermines the size of the sparse matrix. 
%
%   See also SPARSE, ACCUMARRAY
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 14/05/2020. Version: 1.4. Less inputs, Doc improved
%   Created:  10/12/2018. Version: 1.0

%% Assembly of global sparse matrix
if ( strcmp(sets.dTE,'double') && strcmp(sets.dTN,'double') )
    K = sparse(iK, jK, Ke, sets.tdofs, sets.tdofs);
    
elseif ( strcmp(sets.dTE,'uint32') && strcmp(sets.dTN,'double') )
    K = accumarray([iK,jK], Ke, [sets.tdofs sets.tdofs], [], [], 1);
    
else
    error('MATLAB currently does not support "single" data precision for sparse matrices!');
end
