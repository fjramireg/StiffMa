nel = 70;
dTE = 'int32';
dTN = 'single';
[elements, nodes] = CreateMesh(nel,nel,nel,dTE,dTN,0,0);
E = 200e9;
nu = 0.3;
N = size(nodes,1);
[iK, jK] = IndexVectorSymCPUp(elements);
Ke = Hex8vectorSymCPUp(elements,nodes,E,nu);

%% Index computation on CPU (parallel)
[i, j] = IndexVectorSymCPUp(elements);

%% Element stiffness matrices computation on CPU (parallel)
v = Hex8vectorSymCPUp(elements,nodes,E,nu);

