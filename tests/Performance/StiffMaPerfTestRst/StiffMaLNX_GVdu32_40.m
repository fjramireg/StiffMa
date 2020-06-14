sets.nel = 40;
sets.dTE = 'uint32';
sets.dTN = 'double';
[Mesh.elements, Mesh.nodes] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);
sets.nel = 64000;
sets.sz = 300;
sets.edof = 24;
MP.E = 200000000000;
MP.nu = 3.000000e-01;
d = gpuDevice;
sets.tbs = d.MaxThreadsPerBlock;
sets.numSMs   = d.MultiprocessorCount;
sets.WarpSize = d.SIMDWidth;
elementsGPU = gpuArray(Mesh.elements');
nodesGPU = gpuArray(Mesh.nodes');

%% StiffMa-GPU-Vector-Symmetry
[iKd, jKd] = Index_vpsa(elementsGPU, sets);
Ked = eStiff_vpsa(elementsGPU, nodesGPU, MP, sets);
wait(d);
K = AssemblyStiffMa(iKd, jKd, Ked, sets.dTE, sets.dTN);
wait(d);
