function [iK, jK, Ke] = StiffMas4(elements,nodes,c) %#codegen
% STIFFMAS2 Create the global stiffness matrix K for a SCALAR problem in SERIAL computing.
%   STIFFMAS2(elements,nodes,c) returns a sparse matrix K from finite element
%   analysis of scalar problems in a three-dimensional domain, where "elements"
%   is the connectivity matrix of size nelx8, "nodes" the nodal coordinates of
%   size Nx3, and "c" the material property for an isotropic material (scalar).
%
%   See also STIFFMAS
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 	Created: 08/12/2019. Version: 1.0 - The accumarray function is not used

% Add kernelfun pragma to trigger kernel creation
coder.gpu.kernelfun;

% Variable declaration/initialization
p = 1/sqrt(3);                      % Gauss point
r = [-p,p,p,-p,-p,p,p,-p];          % Points through r-coordinate
s = [-p,-p,p,p,-p,-p,p,p];          % Points through s-coordinate
t = [-p,-p,-p,-p,p,p,p,p];          % Points through t-coordinate
dTypeInd = class(elements);         % Data type (precision) for index computation
dTypeKe = class(nodes);             % Data type (precision) for ke computation
nel = size(elements,1);             % Total number of elements
iK = zeros(8,8,nel,dTypeInd);       % Stores the rows' indices
jK = zeros(8,8,nel,dTypeInd);       % Stores the columns' indices
Ke = zeros(8,8,nel,dTypeKe);        % Stores the NNZ values
for e = 1:nel                       % Loop over elements
    n = elements(e,:);              % Nodes of the element 'e'
    X = nodes(n,:);                 % Nodal coordinates of the element 'e'
    jK(:,:,e) = repmat(n,8,1);      % Columm index storage
    iK(:,:,e) = jK(:,:,e)';         % Row index storage
    for i=1:8                       % Loop over numerical integration
        ri = r(i); si = s(i); ti = t(i);
        %  Shape function derivatives with respect to r,s,t. L = [dNdr; dNds; dNdt]; L matrix
        L = [-(1-si)*(1-ti),  (1-si)*(1-ti), (1+si)*(1-ti), -(1+si)*(1-ti),...  % dN/dr;
            -(1-si)*(1+ti),  (1-si)*(1+ti), (1+si)*(1+ti), -(1+si)*(1+ti);
            -(1-ri)*(1-ti), -(1+ri)*(1-ti), (1+ri)*(1-ti),  (1-ri)*(1-ti),...   % dN/ds;
            -(1-ri)*(1+ti), -(1+ri)*(1+ti), (1+ri)*(1+ti),  (1-ri)*(1+ti);
            -(1-ri)*(1-si), -(1+ri)*(1-si),-(1+ri)*(1+si), -(1-ri)*(1+si),...   % dN/dt;
            (1-ri)*(1-si),  (1+ri)*(1-si), (1+ri)*(1+si),  (1-ri)*(1+si)]*(1/8);
        Jac  = L*X;                 % Jacobian matrix
        detJ = det(Jac);            % Jacobian's determinant
        B = Jac\L;                  % B matrix
        Ke(:,:,e) = Ke(:,:,e) + c*detJ*(B'*B);    % Element stiffness matrix - computing & storing
    end
end
