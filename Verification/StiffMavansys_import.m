function K = StiffMavansys_import(folder)
% StiffMavansys_import reads the element stiffness matrices from ANSYS results and
% built the global stiffness matrix.
%
%   Example: K = StiffMansys_import('ANSYS_rst/');
%
%   See also STIFFMASS, STIFFMAPS, SPARSE
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  16/12/2019. Version: 1.0

elements = importElements([folder,'STIFFNESS_VEC.elem']);
nel = size(elements,1);             % Total number of elements
edof= 24;                           % Local number of DOFs
iK  = zeros(edof, edof, nel);       % Stores the rows' indices
jK  = zeros(edof, edof, nel);      	% Stores the columns' indices
Ke  = zeros(edof, edof, nel);      	% Stores the NNZ values
for e = 1:nel                       % Loop over elements
    n = elements(e,1:8);              % Nodes of the element 'e'
    edofs = [3*n-2; 3*n-1; 3*n];    % DOFs of the element 'e'
    ind = repmat(edofs(:),1,edof);  % Index for element 'e'
    iK(:,:,e) = ind;                % Row index storage
    jK(:,:,e) = ind';               % Columm index storage
    name = [folder,'KE',num2str(e),'.dat'];
    Ke(:,:,e) = mm_to_msm(name);    % Element stiffness matrix storage
end
K = accumarray([iK(:),jK(:)],Ke(:),[],[],[],1);% Assembly of the global stiffness matrix
