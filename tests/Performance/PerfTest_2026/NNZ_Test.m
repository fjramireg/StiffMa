sets.nel = 182;
sets.dTE = 'uint32';
sets.dTN = 'double';
[Mesh.elements, Mesh.nodes] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets);
sets.nel = 6028568;
sets.nnodes = 6128487;
sets.sz = 36;
sets.edof = 8;
c = 3.841000e+02;

%% NNZ_CPU_Scalar_double_182
Ke = eStiff_sssa(Mesh, c, sets);
