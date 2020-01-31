function D = DMatrix(E,nu,dType)
% DMATRIX Compute the isotropic material matrix for the VECTOR problem.
%   D = DMATRIX(E,nu,dType) returns the isotropic material matrix "D" of
%   size 6x6 for vector problem in a three-dimensional, where "E" (Young's
%   modulus) and "nu" (Poisson ratio) are the material property for an
%   isotropic material, and "dType" is data type required (single or
%   double). 
%
%   See also STIFFMA_VS, STIFFMA_VSS, STIFFMA_VPS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Modified: 18/12/2019. Version: 1.4. Code modified. Doc improved
% 	Modified: 29/01/2019. Version: 1.3
%   Created:  17/01/2019. Version: 1.0

D = zeros(6,6,dType);               % Initialize D in the correct data type
a = 1 + nu;                         % Constant
b = 1 - 2*nu;                       % Constant
c = 1 - nu;                         % Constant
D(:,:) = [c, nu, nu, 0, 0, 0;       % Fills the matix
          nu, c, nu, 0, 0, 0;
          nu, nu, c, 0, 0, 0;
          0, 0, 0, b/2, 0, 0;
          0, 0, 0, 0, b/2, 0;
          0, 0, 0, 0, 0, b/2]*(E/(a*b));
