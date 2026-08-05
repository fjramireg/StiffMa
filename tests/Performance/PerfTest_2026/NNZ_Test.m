sets.nel = 320;
sets.dTE = 'uint32';
sets.dTN = 'single';
[Mesh.elements, Mesh.nodes] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets);
sets.nel = 32768000;
sets.nnodes = 33076161;
sets.sz = 300;
sets.edof = 24;
MP.E = 200000000000;
MP.nu = 3.000000e-01;

%% NNZ_CPU_Vector_single_320
Ke = eStiff_vssa(Mesh, MP, sets);
