%  Performance test to measure the runtime of the SCALAR problem

% General setup
addpath('../Scalar/');
addpath('../Common');
addpath('../Utils');
nel = 20;               % Number of elements on each direction
dTE = 'int32';         % Data precision for "elements" ['int32', 'uint32', 'int64', 'uint64' or 'double']
dTN = 'double';         % Data precision for "nodes" ['single' or 'double']
[elements, nodes] = CreateMesh(nel,nel,nel,dTE,dTN,0,0); % Mesh creation

%% Index computation
[iK, jK] = IndexScalarSymCPU(elements);     % Row/column indices of tril(K)
