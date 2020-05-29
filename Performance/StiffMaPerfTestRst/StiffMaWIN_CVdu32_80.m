sets.nel = 80;
sets.dTE = 'uint32';
sets.dTN = 'double';
[Mesh.elements, Mesh.nodes] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);
sets.nel = 512000;
sets.sz = 300;
sets.edof = 24;
MP.E = 200000000000;
MP.nu = 3.000000e-01;

%% StiffMa-CPU-Vector
[iK, jK] = Index_va(Mesh.elements', sets);
Ke = eStiff_vsa(Mesh, MP, sets);
K = AssemblyStiffMa(iK, jK, Ke, sets.dTE, sets.dTN);

%% StiffMa-CPU-Vector-Symmetry
[iKs, jKs] = Index_vssa(Mesh.elements, sets);
Kes = eStiff_vssa(Mesh, MP, sets);
Ks = AssemblyStiffMa(iKs, jKs, Kes, sets.dTE, sets.dTN);
