sets.nel = 20;
sets.dTE = 'uint64';
sets.dTN = 'single';
[elements, ~] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);
sets.nel = 8000;
sets.sz = 36;
sets.edof = 8;
[iK, jK] = Index_ssa(elements, sets);
[iK, jK] = Index_sssa(elements, sets);

%% Index-CPU-Scalar
[iK, jK] = Index_ssa(elements, sets);

%% Index-CPU-Scalar-Symmetry
[iK, jK] = Index_sssa(elements, sets);
