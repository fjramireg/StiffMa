sets.nel = 80;
sets.dTE = 'uint32';
sets.dTN = 'single';
[elements, ~] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);
sets.nel = 512000;
sets.sz = 36;
sets.edof = 8;

%% Index-CPU-Scalar
[iK, jK] = Index_ssa(elements, sets);

%% Index-CPU-Scalar-Vectorized
[iK, jK] = Index_sa(elements', sets);

%% Index-CPU-Scalar-Symmetry
[iK, jK] = Index_sssa(elements, sets);
