sets.nel = 370;
sets.dTE = 'uint32';
sets.dTN = 'double';
[Mesh.elements, Mesh.nodes] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);
sets.nel = 50653000;
sets.sz = 36;
sets.edof = 8;
c = 1;
d = gpuDevice;
sets.tbs = d.MaxThreadsPerBlock;
sets.numSMs   = d.MultiprocessorCount;
sets.WarpSize = d.SIMDWidth;
elementsGPU = gpuArray(Mesh.elements');
nodesGPU = gpuArray(Mesh.nodes');

%% EStiff-GPU-Scalar-Symmetry
Ke = eStiff_spsa(elementsGPU, nodesGPU, c, sets);
wait(d);
