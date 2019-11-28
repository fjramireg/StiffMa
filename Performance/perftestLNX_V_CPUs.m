nel = 70;
dTE = 'uint32';
dTN = 'single';
[elements, nodes] = CreateMesh(nel,nel,nel,dTE,dTN,0,0);
E = 200e9;
nu = 0.3;
N = size(nodes,1);
[iK, jK] = IndexVectorSymCPU(elements);
Ke = Hex8vectorSymCPU(elements,nodes,E,nu);

%% Index computation on CPU (serial)
[i, j] = IndexVectorSymCPU(elements);

%% Element stiffness matrices computation on CPU (serial)
v = Hex8vectorSymCPU(elements,nodes,E,nu);

