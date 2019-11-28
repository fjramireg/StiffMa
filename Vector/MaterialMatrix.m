function D = MaterialMatrix(E,nu,dType)
% MATERIALMATRIX Compute the isotropic material matrix for the VECTOR problem
%   MATERIALMATRIX(E,nu,dType) returns the isotropic material matrix "D"
%   for vector problem in a three-dimensional, where "E" (Young's modulus)
%   and "nu" (Poisson ratio) are the material property for an isotropic
%   material, and "dType" is data type required (single or double).
%
%   See also STIFFMATGENVC, STIFFMATGENVCSYMCPU, STIFFMATGENVCSYMCPUP, STIFFMATGENVCSYMGPU
%
%   For more information, see <a href="matlab:
%   web('https://github.com/fjramireg/MatGen')">the MatGen Web site</a>.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created: 17/01/2019. Modified: 29/01/2019. Version: 1.3

D = zeros(6,6,dType);               % Initialize D in the correct data type
D(:,:) = (E/((1 + nu)* (1 - 2*nu)))*...  % Fills the matix
    [1 - nu, nu, nu, 0, 0, 0;
    nu, 1 - nu, nu, 0, 0, 0;
    nu, nu, 1 - nu, 0, 0, 0;
    0, 0, 0, (1 - 2*nu)/2, 0, 0;
    0, 0, 0, 0, (1 - 2*nu)/2, 0;
    0, 0, 0, 0, 0, (1 - 2*nu)/2];
