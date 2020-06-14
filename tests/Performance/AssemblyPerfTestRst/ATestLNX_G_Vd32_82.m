sets.nel = 82;
sets.dTE = 'uint32';
sets.dTN = 'double';
[Mesh.elements, Mesh.nodes] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);
sets.nel = 551368;
sets.sz = 300;
sets.edof = 24;
MP.E = 200000000000;
MP.nu = 3.000000e-01;
d = gpuDevice;
sets.tbs = d.MaxThreadsPerBlock;
sets.numSMs   = d.MultiprocessorCount;
sets.WarpSize = d.SIMDWidth;
[iKd, jKd] = Index_vpsa(gpuArray(Mesh.elements'), sets);
Ked = eStiff_vpsa(gpuArray(Mesh.elements'), gpuArray(Mesh.nodes'), MP, sets);

%% Assembly-GPU-Vector-Symmetry
K = AssemblyStiffMa(iKd, jKd, Ked, sets.dTE, sets.dTN);
wait(d);
