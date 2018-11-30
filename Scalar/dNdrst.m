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

function L = dNdrst
% Computes the shape functions derivatives with respect to r,s,t
p = 1/sqrt(3);              % Gauss points
r = [p,-p,p,-p,p,-p,p,-p];  % Points through r-coordinate
s = [p,p,-p,-p,p,p,-p,-p];  % Points through s-coordinate
t = [p,p,p,p,-p,-p,-p,-p];  % Points through t-coordinate
L = zeros(3,8,8);           % Initialize the matrix L
for i=1:8
    ri = r(i); si = s(i); ti = t(i);
    a = 1-ri; b = 1+ri; c = 1-si; d = 1+si; e = 1-ti; f = 1+ti;
    L(:,:,i) = (0.125)*[-c*e, c*e, d*e, -d*e, -c*f, c*f, d*f, -d*f;... % dNdr
                        -a*e, -b*e, b*e, a*e, -a*f, -b*f, b*f, a*f;... % dNds
                        -a*c, -b*c, -b*d, -a*d, a*c, b*c, b*d, a*d];   % dNdt
end
