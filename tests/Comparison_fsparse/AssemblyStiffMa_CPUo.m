function K = AssemblyStiffMa_CPUo(iK, jK, Ke, tdof)
% ASSEMBLYSTIFFMA_CPUO Assembly a global sparse stiffness matrix K with an optimized CPU code.
%
%   K = ASSEMBLYSTIFFMA_CPUO(iK,jK,Ke,tdof) returns a sparse matrix K that
%   is computed with an optimized CPU code according to the input data,
%   where "iK", "jK", and "Ke" are column vectors containing the row indices,
%   column indices and non-zero value of each entry of the sparse matrix.
%   "tdof" explicitly defines the size of the matrix (total number of
%   degree of freedoms).
%
%   See also SPARSE, ACCUMARRAY
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  June 22, 2020. Version: 1.0
%
% Credits:
%   - The "fsparse" function is used from "stenglib" library developed by:
%   Engblom, S. & Lukarski, D. (2016). Fast MATLAB compatible sparse
%   assembly on multicore computers. Parallel Computing, 56, 1-17.
%   https://doi.org/10.1016/j.parco.2016.04.001.
%   Code: https://github.com/stefanengblom/stenglib
%

% K = sparse(iK, jK, Ke, tdof , tdof);
K = fsparse(iK, jK, Ke, [tdof , tdof]);
