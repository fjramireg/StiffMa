sets.nel = 80;
sets.dTE = 'uint32';
sets.dTN = 'double';
[Mesh.elements, Mesh.nodes] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);
sets.nel = 512000;
sets.sz = 36;
sets.edof = 8;
c = 1;

%% EStiff-CPU-Scalar
Ke = eStiff_ssa(Mesh, c, sets);

%% EStiff-CPU-Scalar-Symmetry
Ke = eStiff_sssa(Mesh, c, sets);
