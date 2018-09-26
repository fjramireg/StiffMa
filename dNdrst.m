function L = dNdrst
% Computes and stores the shape functions derivatives with respect to r,s,t

% Gauss points table
p = 1/sqrt(3);
r = [p,-p,p,-p,p,-p,p,-p];
s = [p,p,-p,-p,p,p,-p,-p];
t = [p,p,p,p,-p,-p,-p,-p];

% Shape functions derivatives
L = zeros(3,8,8);
for i=1:8
    ri = r(i); si = s(i); ti = t(i);
    L(:,:,i) = (0.125)*[...
        -(1-si)*(1-ti),  (1-si)*(1-ti), (1+si)*(1-ti), -(1+si)*(1-ti),...
        -(1-si)*(1+ti),  (1-si)*(1+ti), (1+si)*(1+ti), -(1+si)*(1+ti);...
        -(1-ri)*(1-ti), -(1+ri)*(1-ti), (1+ri)*(1-ti),  (1-ri)*(1-ti),...
        -(1-ri)*(1+ti), -(1+ri)*(1+ti), (1+ri)*(1+ti),  (1-ri)*(1+ti);...
        -(1-ri)*(1-si), -(1+ri)*(1-si),-(1+ri)*(1+si), -(1-ri)*(1+si),...
        (1-ri)*(1-si),  (1+ri)*(1-si), (1+ri)*(1+si),  (1-ri)*(1+si)];
end
