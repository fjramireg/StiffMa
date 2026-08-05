sets.nel = 364;
sets.dTE = 'uint64';
sets.dTN = 'single';
[elements, ~] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets);
sets.nel = 48228544;
sets.edof = 24;
sets.sz = 300;

%% Index_CPU_Vector_uint64_364
[iK, jK] = Index_vssa(elements, sets);
