nel = 70;
dTE = 'int32';
dTN = 'single';
[elements, nodes] = CreateMesh(nel,nel,nel,dTE,dTN,0,0);
E = 200e9;
nu = 0.3;
N = size(nodes,1);
elementsGPU = gpuArray(elements');
nodesGPU    = gpuArray(nodes');
[iK, jK] = IndexVectorSymGPU(elementsGPU);
Ke = Hex8vectorSymGPU(elementsGPU,nodesGPU,E,nu);

%% Transfer to GPU memory
eGPU = gpuArray(elements');
nGPU = gpuArray(nodes');

%% Index computation on GPU
[i, j] = IndexVectorSymGPU(elementsGPU);

%% Element stiffness matrices computation on GPU
v = Hex8vectorSymGPU(elementsGPU,nodesGPU,E,nu);

