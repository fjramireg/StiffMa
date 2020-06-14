function L = dNdrst(dType)
% DNDRST Compute the shape functions derivatives with respect to r,s,t (Hex8)
%   DNDRST(dType) Returns a matrix "L" of size [3*8*8] with all shape
%   function derivatives in natural coordinates. As input it requieres the
%   data type "dType" as single or double
%
%   See also HEX8SCALAR, HEX8VECTOR
%
%   For more information, see <a href="matlab:
%   web('https://github.com/fjramireg/MatGen')">the MatGen Web site</a>.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created: 30/11/2018. 
%	Modified: 03/12/2019. Version: 1.4. Change the Gauss table
%	Modified: 21/01/2019. Version: 1.3

p = zeros(1,dType);         % Initialize p in the correct data type
p(1) = 1/sqrt(3);           % Gauss point value
r = [-p,p,p,-p,-p,p,p,-p];  % Points through r-coordinate
s = [-p,-p,p,p,-p,-p,p,p];  % Points through s-coordinate
t = [-p,-p,-p,-p,p,p,p,p];  % Points through t-coordinate
L = zeros(3,8,8,dType);     % Initialize the matrix L
for i=1:8
    ri = r(i); si = s(i); ti = t(i);
    a = 1-ri; b = 1+ri; c = 1-si; d = 1+si; e = 1-ti; f = 1+ti;
    L(:,:,i) = [-c*e, c*e, d*e, -d*e, -c*f, c*f, d*f, -d*f;... % dN/dr
                -a*e, -b*e, b*e, a*e, -a*f, -b*f, b*f, a*f;... % dN/ds
                -a*c, -b*c, -b*d, -a*d, a*c, b*c, b*d, a*d]/8; % dN/dt
end
