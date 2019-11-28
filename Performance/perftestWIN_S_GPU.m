nel = 70;
dTE = 'int32';
dTN = 'single';
[elements, nodes] = CreateMesh(nel,nel,nel,dTE,dTN,0,0);
c = 1;
N = size(nodes,1);
elementsGPU = gpuArray(elements');
nodesGPU    = gpuArray(nodes');
[iK, jK] = IndexScalarSymGPU(elementsGPU);
Ke = Hex8scalarSymGPU(elementsGPU,nodesGPU,c);

%% Transfer to GPU memory
eGPU = gpuArray(elements');
nGPU = gpuArray(nodes');

%% Index computation on GPU
[i, j] = IndexScalarSymGPU(elementsGPU);

%% Element stiffness matrices computation on GPU
v = Hex8scalarSymGPU(elementsGPU,nodesGPU,c);

