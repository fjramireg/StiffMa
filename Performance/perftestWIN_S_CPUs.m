nel = 10;
dTE = 'uint32';
dTN = 'single';
[elements, nodes] = CreateMesh(nel,nel,nel,dTE,dTN,0,0);
c = 1;
N = size(nodes,1);
[iK, jK] = IndexScalarSymCPU(elements);
Ke = Hex8scalarSymCPU(elements,nodes,c);

%% Index computation on CPU (serial)
[i, j] = IndexScalarSymCPU(elements);

%% Element stiffness matrices computation on CPU (serial)
v = Hex8scalarSymCPU(elements,nodes,c);

