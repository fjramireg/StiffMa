sets.nel = 10;
sets.dTE = 'uint32';
sets.dTN = 'double';
[Mesh.elements, Mesh.nodes] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);
sets.nel = 1000;
sets.sz = 300;
sets.edof = 24;
MP.E = 200000000000;
MP.nu = 3.000000e-01;
[iK, jK] = Index_va(Mesh.elements', sets);
Ke = eStiff_vsa(Mesh, MP, sets);
[iKs, jKs] = Index_vssa(Mesh.elements, sets);
Kes = eStiff_vssa(Mesh, MP, sets);

%% Assembly-CPU-Vector
K = AssemblyStiffMa(iK, jK, Ke, sets.dTE, sets.dTN);

%% Assembly-CPU-Vector-Symmetry
Ks = AssemblyStiffMa(iKs, jKs, Kes, sets.dTE, sets.dTN);
