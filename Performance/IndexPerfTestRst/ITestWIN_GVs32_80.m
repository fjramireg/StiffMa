sets.nel = 80;
sets.dTE = 'uint32';
sets.dTN = 'single';
[elements, ~] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);
sets.nel = 512000;
sets.edof = 24;
sets.sz = 300;
d = gpuDevice;
sets.tbs = d.MaxThreadsPerBlock;
sets.numSMs   = d.MultiprocessorCount;
sets.WarpSize = d.SIMDWidth;
elementsGPU = gpuArray(elements');

%% Index-GPU-Vector-Symmetry
[iKd, jKd] = Index_vpsa(elementsGPU, sets);
wait(d);
