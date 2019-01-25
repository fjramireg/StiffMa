%  Performance test to measure the runtime of the SCALAR problem 

% General setup
addpath('../Scalar/');
addpath('../Common');
addpath('../Utils');
nel = 10;               % Number of elements on each direction
dTE = 'int32';         % Data precision for "elements" ['int32', 'uint32', 'int64', 'uint64' or 'double']
dTN = 'double';         % Data precision for "nodes" ['single' or 'double']
[elements, nodes] = CreateMesh(nel,nel,nel,dTE,dTN,0,0); % Mesh creation
c = 1;                  % Material properties
N = size(nodes,1);      % Total number of nodes (DOFs)
[iK, jK] = IndexScalarSymCPU(elements);     % Row/column indices of tril(K)
Ke = Hex8scalarSymCPU(elements,nodes,c);    % Entries of tril(K)
name = ['CPU','_N',dTN,'_E',dTE,'_nel',num2str(nel)];

%% Index computation
[iK, jK] = IndexScalarSymCPU(elements);     % Row/column indices of tril(K)

%% Element stiffness matrices computation
Ke = Hex8scalarSymCPU(elements,nodes,c);    % Entries of tril(K)

%% Assembly of global sparse matrix on CPU
K = AssemblyStiffMat(iK,jK,Ke(:),N,dTE,dTN);% Triangular sparse matrix
