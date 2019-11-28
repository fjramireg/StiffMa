nel = 70;
dTE = 'int32';
dTN = 'single';
[elements, nodes] = CreateMesh(nel,nel,nel,dTE,dTN,0,0);
c = 1;
N = size(nodes,1);
[iK, jK] = IndexScalarSymCPUp(elements);
Ke = Hex8scalarSymCPUp(elements,nodes,c);

%% Index computation on CPU (parallel)
[i, j] = IndexScalarSymCPU(elements);

%% Element stiffness matrices computation on CPU (parallel)
v = Hex8scalarSymCPU(elements,nodes,c);

