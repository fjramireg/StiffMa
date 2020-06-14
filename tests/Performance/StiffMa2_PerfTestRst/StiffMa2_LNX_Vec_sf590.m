% Inputs
nel = 90;
sets.sf = 5;
sets.dTE = 'uint32';
sets.dTN = 'double';
MP.c = 384.1;
MP.E = 200e9;
MP.nu = 0.3;

% Mesh generation
[Mesh.elements, Mesh.nodes] = CreateMesh2(nel, nel, nel, sets.dTE, sets.dTN);
[sets.nel, sets.nxe]  = size(Mesh.elements);
[sets.nnod, sets.dim] = size(Mesh.nodes);
sets.dxn = 3;
sets.edof = sets.dxn * sets.nxe;
sets.sz = (sets.edof * (sets.edof + 1) )/2;
sets.tdofs = sets.nnod * sets.dxn;

% GPU setup
dev = gpuDevice;
sets.tbs = dev.MaxThreadsPerBlock;
sets.numSMs   = dev.MultiprocessorCount;
sets.WarpSize = dev.SIMDWidth;

% Number of chuncks
d_et  = zeros(1,1,sets.dTE);
d_et1 = whos('d_et');
szInd = d_et1.bytes;
d_nt  = zeros(1,1,sets.dTN);
d_nt1 = whos('d_nt');
szNNZ = d_nt1.bytes;
Mmesh  = szInd*numel(Mesh.elements) + szNNZ*numel(Mesh.nodes);
Mtrip  = (2*szInd + szNNZ)*sets.sz*sets.nel;
Maccum = 3*Mtrip;
Mcsc   = 0.5*Mtrip;
Mtotal = Mmesh + Mtrip + Maccum + Mcsc;
ndiv = ceil(Mtotal/dev.AvailableMemory);
ndiv = ndiv + sets.sf;
while mod(sets.nel,ndiv) ~= 0
    ndiv = ndiv + 1;
end

% Available memory check
Mtotal_c = Mmesh + Mtrip*(4/ndiv + 1/2);
if Mtotal_c > dev.AvailableMemory
    reset(dev);
    x =['No enough memory on the GPU to process the mesh with ', num2str(ndiv), ' chunks'];
    error(x);
else
    x=['The global stiffness matrix will be computed with ', num2str(ndiv), ' chunk(s).'];
    disp(x);
end

% Transfer memory: host to device
x=['Available memory on GPU before computations begin (MB): ', num2str(dev.AvailableMemory/1e6)];
disp(x);
elementsGPU = gpuArray(Mesh.elements');
nodesGPU = gpuArray(Mesh.nodes');
x=['Processing the VECTOR problem with ', num2str(nel),'x',num2str(nel),'x',num2str(nel), ' elements'];
disp(x);

%% StiffMa
if (ndiv > 1)
    m = sets.nel / ndiv;
    sets.nel = m;
    K = sparse(sets.tdofs, sets.tdofs);
    for i=1:ndiv
        ini = 1 + m*(i-1);
        fin = ini + m - 1;
        [iKd, jKd] = Index_vpsa(elementsGPU(:, ini:fin), sets);
        Ked = eStiff_vpsa(elementsGPU(:, ini:fin), nodesGPU, MP, sets);
        wait(dev);
        K = K + AssemblyStiffMa(iKd, jKd, Ked, sets);
    end
    clear elementsGPU nodesGPU iKd jKd Ked
else
    [iKd, jKd] = Index_vpsa(elementsGPU, sets);
    Ked = eStiff_vpsa(elementsGPU, nodesGPU, MP, sets);
    wait(dev);
    clear elementsGPU nodesGPU
    K = AssemblyStiffMa(iKd, jKd, Ked, sets);
    clear iKd jKd Ked
end
wait(dev);
