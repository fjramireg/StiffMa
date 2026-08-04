sets.nel = 160;
sets.dTE = 'uint32';
sets.dTN = 'double';
[Mesh.elements, Mesh.nodes] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets);
sets.nel = 4096000;
sets.nnodes = 4173281;
sets.sz = 300;
sets.edof = 24;
MP.E = 200000000000;
MP.nu = 3.000000e-01;

%% NNZ_CPU_Vector_double_160
Ke = eStiff_vssa(Mesh, MP, sets);
