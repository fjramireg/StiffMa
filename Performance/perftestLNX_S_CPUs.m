nel = 100;
dTE = 'double';
dTN = 'double';
[elements, nodes] = CreateMesh(nel,nel,nel,dTE,dTN,0,0);
c = 1;
N = size(nodes,1);
[iK, jK] = IndexScalarSymCPU(elements);
Ke = Hex8scalarSymCPU(elements,nodes,c);

%% Index computation on CPU (serial)
[i, j] = IndexScalarSymCPU(elements);

%% Element stiffness matrices computation on CPU (serial)
v = Hex8scalarSymCPU(elements,nodes,c);

%% Assembly of global sparse matrix on CPU
K = AssemblyStiffMat(iK,jK,Ke,N,dTE,dTN);
