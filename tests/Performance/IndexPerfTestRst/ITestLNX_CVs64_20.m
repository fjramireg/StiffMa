sets.nel = 20;
sets.dTE = 'uint64';
sets.dTN = 'single';
[elements, ~] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);
sets.nel = 8000;
sets.edof = 24;
sets.sz = 300;
[iK, jK] = Index_vsa(elements, sets);
[iK, jK] = Index_vssa(elements, sets);

%% Index-CPU-Vector
[iK, jK] = Index_vsa(elements, sets);

%% Index-CPU-Vector-Symmetry
[iK, jK] = Index_vssa(elements, sets);
