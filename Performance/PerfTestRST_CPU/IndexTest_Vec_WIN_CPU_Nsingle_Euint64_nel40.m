sets.nel = 40;
sets.dTE = 'uint64';
sets.dTN = 'single';
[elements, ~] = CreateMesh(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);
sets.sz = 300;
sets.edof = 24;

%% Index computation on CPU (Vector)
[iK, jK] = Index_vss(elements, sets);
