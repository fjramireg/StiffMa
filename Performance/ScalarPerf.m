%  Performance measure of the SCALAR problem

%% General setup
addpath('../Scalar/');
addpath('../Common');
addpath('../Utils');
nel = 10;            % Number of elements on each direction
dTypeE = 'uint32';  % Data precision for "elements" ['int32', 'uint32', 'int64', 'uint64' or 'double']
dTypeN = 'double';  % Data precision for "nodes" ['single' or 'double']
[elements, nodes] = CreateMesh(nel,nel,nel,dTypeE,dTypeN,0,0); % Mesh creation
c = 1;              % Material properties

%% Timing the execution of CPU serial code

% Index creation
fh_ind_h = @() IndexScalarSymCPU(elements);    % Handle to function Index on CPU
rt_ind_h = timeit(fh_ind_h,2)              % timing with 2 outputs

% Element stiffness matrices computation
fh_Ke_h = @() Hex8scalarSymCPU(elements,nodes,c);    
rt_Ke_h = timeit(fh_Ke_h,1)

% Assembly of global stiffness matrix
[iK, jK] = IndexScalarSymCPU(elements); 
Ke = Hex8scalarSymCPU(elements,nodes,c);
N = size(nodes,1);                      % Total number of nodes (DOFs)
dTE = class(elements);                  % "elements" data precision
dTN = class(nodes);                     % "nodes" data precision
fh_Kg_h = @()  AssemblyStiffMat(iK,jK,Ke(:),N,dTE,dTN);
rt_Kg_h = timeit(fh_Kg_h,1)
