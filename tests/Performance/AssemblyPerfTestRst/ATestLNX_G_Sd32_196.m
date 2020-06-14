sets.nel = 196;
sets.dTE = 'uint32';
sets.dTN = 'double';
[Mesh.elements, Mesh.nodes] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);
sets.nel = 7529536;
sets.sz = 36;
sets.edof = 8;
c = 3.841000e+02;
d = gpuDevice;
sets.tbs = d.MaxThreadsPerBlock;
sets.numSMs   = d.MultiprocessorCount;
sets.WarpSize = d.SIMDWidth;
[iKd, jKd] = Index_spsa(gpuArray(Mesh.elements'), sets);
Ked = eStiff_spsa(gpuArray(Mesh.elements'), gpuArray(Mesh.nodes'), c, sets);

%% Assembly-GPU-Scalar-Symmetry
K = AssemblyStiffMa(iKd, jKd, Ked, sets.dTE, sets.dTN);
wait(d);
