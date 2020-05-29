sets.nel = 160;
sets.dTE = 'uint32';
sets.dTN = 'double';
[Mesh.elements, Mesh.nodes] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);
sets.nel = 4096000;
sets.sz = 36;
sets.edof = 8;
c = 3.841000e+02;
d = gpuDevice;
sets.tbs = d.MaxThreadsPerBlock;
sets.numSMs   = d.MultiprocessorCount;
sets.WarpSize = d.SIMDWidth;
elementsGPU = gpuArray(Mesh.elements');
nodesGPU = gpuArray(Mesh.nodes');

%% StiffMa-GPU-Scalar-Symmetry
[iKd, jKd] = Index_spsa(elementsGPU, sets);
Ked = eStiff_spsa(elementsGPU, nodesGPU, c, sets);
wait(d);
K = AssemblyStiffMa(iKd, jKd, Ked, sets.dTE, sets.dTN);
wait(d);
