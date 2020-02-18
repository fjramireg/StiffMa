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
sets.sz = 36;
sets.edof = 8;
c = 1;

%% EStiff-CPU-Scalar
Ke = eStiff_ssa(Mesh, c, sets);

%% EStiff-CPU-Scalar-Symmetry
Ke = eStiff_sssa(Mesh, c, sets);

%% EStiff-GPU-Scalar-Symmetry
Ke = eStiff_spsa(elementsGPU, nodesGPU, c, sets);
wait(d);