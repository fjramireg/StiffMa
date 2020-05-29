sets.nel = 135;
Case = 'Old';
sets.dTE = 'uint32';
sets.dTN = 'double';
[Mesh.elements, Mesh.nodes] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);
sets.nel = sets.nel^3;
sets.sz = 36;
sets.edof = 8;
sets.tdofs = size(Mesh.nodes,1)*1;   % New
c = 3.841000e+02;
d = gpuDevice;
sets.tbs = d.MaxThreadsPerBlock;
sets.numSMs   = d.MultiprocessorCount;
sets.WarpSize = d.SIMDWidth;
elementsGPU = gpuArray(Mesh.elements');
nodesGPU = gpuArray(Mesh.nodes');

if strcmp(Case,'New')
    %% StiffMa-GPU-Scalar-Symmetry -- New
    subs = 4;
    nent = sets.nel * sets.sz;
    m = nent / subs;
    n = subs;
    [iKd, jKd] = Index_spsa(elementsGPU, sets);
    Ked = eStiff_spsa(elementsGPU, nodesGPU, c, sets);
    wait(d);
    % New
    iKd = reshape(iKd, m, n);
    jKd = reshape(jKd, m, n);
    Ked = reshape(Ked, m, n);
    K = sparse(sets.tdofs, sets.tdofs);
    for i=1:subs
        K = K + AssemblyStiffMa(iKd(:,i), jKd(:,i), Ked(:,i), sets);  % New
    end
    wait(d);
    
elseif strcmp(Case,'Old')
    %% StiffMa-GPU-Scalar-Symmetry -- Old
    [iKd, jKd] = Index_spsa(elementsGPU, sets);
    Ked = eStiff_spsa(elementsGPU, nodesGPU, c, sets);
    wait(d);
    K = AssemblyStiffMa(iKd, jKd, Ked, sets);
    wait(d);
    
end

%% Clear GPU memory
reset(gpuDevice);
