% Inputs
nel = 140;
sets.sf = 1;
sets.dTE = 'uint32';
sets.dTN = 'double';
MP.E = 200e9;
MP.nu = 0.3;
ct = 384.1;

% Mesh generation
[elements, nodes] = CreateMesh2(nel, nel, nel, sets);
[sets.nel, sets.nxe]  = size(elements);
[sets.nnod, sets.dim] = size(nodes);

% GPU setup
dev = gpuDevice;
sets.tbs = dev.MaxThreadsPerBlock;
sets.numSMs   = dev.MultiprocessorCount;
sets.WarpSize = dev.SIMDWidth;

% GPU Memory transfer
elementsGPU = gpuArray(elements');
nodesGPU = gpuArray(nodes');
sets.dxn = 1;
sets.edof = sets.dxn * sets.nxe;
sets.sz = (sets.edof * (sets.edof + 1) )/2;
sets.tdofs = sets.nnod * sets.dxn;

% For GPU assembly
[iKd, jKd] = Index_spsa(elementsGPU, sets);
Ked = eStiff_spsa(elementsGPU, nodesGPU, ct, sets);

%% Index GPU
[iK, jK] = Index_spsa(elementsGPU, sets);
wait(dev);

%% Local ke GPU
Ke = eStiff_spsa(elementsGPU, nodesGPU, ct, sets);
wait(dev);

%% Assembly GPU
K = AssemblyStiffMa(iKd, jKd, Ked, sets);
wait(dev);
