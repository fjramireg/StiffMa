sets.nel = 320;
sets.dTE = 'uint64';
sets.dTN = 'single';
[elements, ~] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);
sets.nel = 32768000;
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
