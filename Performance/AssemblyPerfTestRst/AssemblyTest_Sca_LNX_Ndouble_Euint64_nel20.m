sets.nel = 20;
sets.dTE = 'uint64';
sets.dTN = 'double';
[Mesh.elements, Mesh.nodes] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);
sets.nel = 8000;
d = gpuDevice;
sets.tbs = d.MaxThreadsPerBlock;
sets.numSMs   = d.MultiprocessorCount;
sets.WarpSize = d.SIMDWidth;
elementsGPU = gpuArray(Mesh.elements');
nodesGPU = gpuArray(Mesh.nodes');
sets.sz = 36;
sets.edof = 8;
c = 1;
[iKd, jKd] = Index_spsa(elementsGPU, sets);
Ked = eStiff_spsa(elementsGPU, nodesGPU, c, sets);
clear elementsGPU nodesGPU;
iKs = gather(iKd);
jKs = gather(jKd);
Kes = gather(Ked);
[iK, jK] = Index_ssa(Mesh.elements, sets);
Ke = eStiff_ssa(Mesh, c, sets);

%% Assembly-CPU-Scalar
K = AssemblyStiffMa(iK, jK, Ke, sets.dTE, sets.dTN);

%% Assembly-CPU-Scalar-Symmetry
K = AssemblyStiffMa(iKs, jKs, Kes, sets.dTE, sets.dTN);

%% Assembly-GPU-Scalar-Symmetry
K = AssemblyStiffMa(iKd, jKd, Ked, sets.dTE, sets.dTN);
wait(d);
