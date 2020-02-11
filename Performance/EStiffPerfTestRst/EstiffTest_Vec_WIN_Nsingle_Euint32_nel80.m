sets.nel = 80;
sets.dTE = 'uint32';
sets.dTN = 'single';
[Mesh.elements, Mesh.nodes] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);
sets.nel = 512000;
d = gpuDevice;
sets.tbs = d.MaxThreadsPerBlock;
sets.numSMs   = d.MultiprocessorCount;
sets.WarpSize = d.SIMDWidth;
elementsGPU = gpuArray(Mesh.elements');
nodesGPU = gpuArray(Mesh.nodes');
sets.sz = 300;
sets.edof = 24;
MP.E = 200000000000;
MP.nu = 3.000000e-01;

%% EStiff-CPU-Vector
% Ke = eStiff_vsa(Mesh, MP, sets);
% 
% %% EStiff-CPU-Vector-Symmetry
% Ke = eStiff_vssa(Mesh, MP, sets);

%% EStiff-GPU-Vector-Symmetry
Ke = eStiff_vpsa(elementsGPU, nodesGPU, MP, sets);
wait(d);
