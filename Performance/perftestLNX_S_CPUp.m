nel = 100;
dTE = 'double';
dTN = 'double';
[elements, nodes] = CreateMesh(nel,nel,nel,dTE,dTN,0,0);
c = 1;
N = size(nodes,1);
[iK, jK] = IndexScalarSymCPUp(elements);
Ke = Hex8scalarSymCPUp(elements,nodes,c);

%% Index computation on CPU (parallel)
[i, j] = IndexScalarSymCPUp(elements);

%% Element stiffness matrices computation on CPU (parallel)
v = Hex8scalarSymCPUp(elements,nodes,c);

%% Assembly of global sparse matrix on CPU
K = AssemblyStiffMat(iK,jK,Ke,N,dTE,dTN);
