sets.nel = 10;
sets.dTE = 'uint32';
sets.dTN = 'single';
[Mesh.elements, Mesh.nodes] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);
sets.nel = 1000;
sets.sz = 300;
sets.edof = 24;
MP.E = 200000000000;
MP.nu = 3.000000e-01;
Ke = eStiff_vsa(Mesh, MP, sets);
Ke = eStiff_vssa(Mesh, MP, sets);

%% EStiff-CPU-Vector
Ke = eStiff_vsa(Mesh, MP, sets);

%% EStiff-CPU-Vector-Symmetry
Ke = eStiff_vssa(Mesh, MP, sets);
