sets.nel = 80;
sets.dTE = 'uint32';
sets.dTN = 'double';
[Mesh.elements, Mesh.nodes] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);
sets.nel = 512000;
sets.sz = 36;
sets.edof = 8;
c = 1;
[iK, jK] = Index_sa(Mesh.elements', sets);
Ke = eStiff_ssa(Mesh, c, sets);
[iKs, jKs] = Index_sssa(Mesh.elements, sets);
Kes = eStiff_sssa(Mesh, c, sets);

%% Assembly-CPU-Scalar
K = AssemblyStiffMa(iK, jK, Ke, sets.dTE, sets.dTN);

%% Assembly-CPU-Scalar-Symmetry
Ks = AssemblyStiffMa(iKs, jKs, Kes, sets.dTE, sets.dTN);
