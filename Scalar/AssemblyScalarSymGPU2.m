%  * ====================================================================*/
% ** This function was developed by:
%  *          Francisco Javier Ramirez-Gil
%  *          Universidad Nacional de Colombia - Medellin
%  *          Department of Mechanical Engineering
%  *
%  ** Please cite this code as:
%  *
%  ** Date & version
%  *      30/11/2018.
%  *      V 1.2
%  *
%  * ====================================================================*/

function K = AssemblyScalarSymGPU2(elements,nodes,c)
% Construction of the global stiffness matrix K (SCALAR-SYMMETRIC-DOUBLE-ACCUMARRAY)
N = size(nodes,1);                              % Total number of nodes
[iK, jK] = IndexScalar(elements);               % Row/column indices of tril(K)
Ke = Hex8scalarSymGPU(elements,nodes,c);        % Entries of tril(K)
K = sparse(double(iK), double(jK), Ke, N, N);   % Assembly of global K
