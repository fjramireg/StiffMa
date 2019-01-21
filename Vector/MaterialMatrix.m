%  * ====================================================================*/
% ** This function was developed by:
%  *          Francisco Javier Ramirez-Gil
%  *          Universidad Nacional de Colombia - Medellin
%  *          Department of Mechanical Engineering
%  *
%  ** Please cite this code as:
%  *
%  ** Date & version
%  *      Created: 16/01/2019. Last modified: 21/01/2019
%  *      V 1.3
%  *
%  * ====================================================================*/

function D = MaterialMatrix(E,nu,dType)
% Isotropic material matrix for the VECTOR problem
D = zeros(6,6,dType);               % Initialize D in the correct data type
D(:,:) = (E/((1 + nu)* (1 - 2*nu)))*...  % Fills the matix
    [1 - nu, nu, nu, 0, 0, 0;
    nu, 1 - nu, nu, 0, 0, 0;
    nu, nu, 1 - nu, 0, 0, 0;
    0, 0, 0, (1 - 2*nu)/2, 0, 0;
    0, 0, 0, 0, (1 - 2*nu)/2, 0;
    0, 0, 0, 0, 0, (1 - 2*nu)/2];
