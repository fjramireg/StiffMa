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
sets.sz = 300;
sets.edof = 24;
MP.E = 200000000000;
MP.nu = 3.000000e-01;
[iKd, jKd] = Index_vpsa(elementsGPU, sets);
Ked = eStiff_vpsa(elementsGPU, nodesGPU, MP, sets);
iKs = gather(iKd);
jKs = gather(jKd);
Kes = gather(Ked);
[iK, jK] = Index_vsa(Mesh.elements, sets);
Ke = eStiff_vsa(Mesh, MP, sets);

%% Assembly-CPU-Vector
K = AssemblyStiffMa(iK, jK, Ke, sets.dTE, sets.dTN);

%% Assembly-CPU-Vector-Symmetry
K = AssemblyStiffMa(iKs, jKs, Kes, sets.dTE, sets.dTN);

%% Assembly-GPU-Vector-Symmetry
K = AssemblyStiffMa(iKd, jKd, Ked, sets.dTE, sets.dTN);
wait(d);
