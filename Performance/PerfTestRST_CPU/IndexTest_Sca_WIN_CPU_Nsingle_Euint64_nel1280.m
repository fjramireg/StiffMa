sets.nel = 1280;
sets.dTE = 'uint64';
sets.dTN = 'single';
[elements, ~] = CreateMesh(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);
sets.sz = 36;

%% Index computation on CPU (Scalar)
[iK, jK] = Index_sss(elements, sets);
