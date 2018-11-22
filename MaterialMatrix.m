function D = MaterialMatrix(E,u)
% Isotropic material matrix for the VECTOR problem
a = 1 + u;
b = 1 - 2*u;
c = 1 - u;
d = E/(a*b);
D = d*[c u u 0 0 0;
       u c u 0 0 0;
       u u c 0 0 0;
       0 0 0 b/2 0 0;
       0 0 0 0 b/2 0;
       0 0 0 0 0 b/2];
