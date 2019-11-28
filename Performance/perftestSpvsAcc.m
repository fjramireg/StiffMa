nel = 200;
dTE = 'double';
dTN = 'double';
[elements, nodes] = CreateMesh(nel,nel,nel,dTE,dTN,0,0);
E = 200e9;
nu = 0.3;
N = size(nodes,1);
[iK, jK] = IndexVectorSymGPU(gpuArray(elements'));
Ke = Hex8vectorSymGPU(gpuArray(elements'),gpuArray(nodes'),E,nu);

%% Assembly of K (vector) using sparse on GPU
K = sparse(iK, jK, Ke, 3*N, 3*N);

%% Assembly of K (vector) using accumarray on GPU
K = accumarray([iK,jK], Ke, [3*N,3*N], [], [], 1);
%% Reset GPU device
reset(gpuDevice);