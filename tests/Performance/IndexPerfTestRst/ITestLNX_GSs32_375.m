sets.nel = 375;
sets.dTE = 'uint32';
sets.dTN = 'single';
[elements, ~] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);
sets.nel = 52734375;
sets.sz = 36;
sets.edof = 8;
d = gpuDevice;
sets.tbs = d.MaxThreadsPerBlock;
sets.numSMs   = d.MultiprocessorCount;
sets.WarpSize = d.SIMDWidth;
elementsGPU = gpuArray(elements');

%% Index-GPU-Scalar-Symmetry
[iKd, jKd] = Index_spsa(elementsGPU, sets);
wait(d);
