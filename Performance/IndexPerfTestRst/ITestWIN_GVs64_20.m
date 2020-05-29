sets.nel = 20;
sets.dTE = 'uint64';
sets.dTN = 'single';
[elements, ~] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);
sets.nel = 8000;
sets.edof = 24;
sets.sz = 300;
d = gpuDevice;
sets.tbs = d.MaxThreadsPerBlock;
sets.numSMs   = d.MultiprocessorCount;
sets.WarpSize = d.SIMDWidth;
elementsGPU = gpuArray(elements');
[iKd, jKd] = Index_vpsa(elementsGPU, sets);

%% Index-GPU-Vector-Symmetry
[iKd, jKd] = Index_vpsa(elementsGPU, sets);
wait(d);
