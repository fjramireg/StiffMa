% Script to test the StiffMa code by dividing [iK, jK, Ke] arrays

sets.nel = 80;
ndiv = 4;               % Number of divisions 
Problem = 'Vector';     % 'Scalar' or 'Vector'
sets.dTE = 'uint32';
sets.dTN = 'double';

%% Adding folders to the path
addpath('../Scalar/');
addpath('../Vector/');
addpath('../Common');
addpath('../Utils');
addpath(pwd);

%% Mesh creation
[Mesh.elements, Mesh.nodes] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);
sets.nel = sets.nel^3;

%% GPU setup
d = gpuDevice;
sets.tbs = d.MaxThreadsPerBlock;
sets.numSMs   = d.MultiprocessorCount;
sets.WarpSize = d.SIMDWidth;

%% Transfer memory: host to device
fprintf('\t Available Memory on GPU before start:           \t %G MB \n', d.AvailableMemory/1e6);
elementsGPU = gpuArray(Mesh.elements');
nodesGPU = gpuArray(Mesh.nodes');
fprintf('\t Available Memory on GPU after mesh transfer:    \t %G MB \n', d.AvailableMemory/1e6);

%% Stiffness matrix generation

% Scalar
if strcmp(Problem,'Scalar')
    sets.sz = 36;
    sets.edof = 8;
    sets.tdofs = size(Mesh.nodes,1)*1;   % New
    c = 3.841000e+02;
    
    [iKd, jKd] = Index_spsa(elementsGPU, sets);
    fprintf('\t Available Memory on GPU after INDEX creation:   \t %G MB \n', d.AvailableMemory/1e6);
    Ked = eStiff_spsa(elementsGPU, nodesGPU, c, sets);
    fprintf('\t Available Memory on GPU after KE creation:      \t %G MB \n', d.AvailableMemory/1e6);
    clear elementsGPU nodesGPU
    wait(d);
    fprintf('\t Available Memory on GPU after mesh deletion:    \t %G MB \n', d.AvailableMemory/1e6);
    
    if (ndiv > 1)
        %  New
        nent = sets.nel * sets.sz;
        m = nent / ndiv;
        n = ndiv;
        iKd = reshape(iKd, m, n);
        jKd = reshape(jKd, m, n);
        Ked = reshape(Ked, m, n);
        K = sparse(sets.tdofs, sets.tdofs);
        for i=1:ndiv
            K = K + AssemblyStiffMa(iKd(:,1), jKd(:,1), Ked(:,1), sets);  % New
            fprintf('\t Available Memory on GPU after ASSEMBLY:         \t %G MB \n', d.AvailableMemory/1e6);
            iKd(:,1) = []; jKd(:,1) = []; Ked(:,1) = [];
            fprintf('\t Available Memory on GPU after DELETE:           \t %G MB \n', d.AvailableMemory/1e6);
        end
        wait(d);
        
    else
        % Old
        K = AssemblyStiffMa(iKd, jKd, Ked, sets);
        wait(d);
        
    end
    
    
    %% Vector
elseif strcmp(Problem,'Vector')
    sets.sz = 300;
    sets.edof = 24;
    sets.tdofs = size(Mesh.nodes,1)*3;   % New
    MP.E = 200000000000;
    MP.nu = 3.000000e-01;
    
    [iKd, jKd] = Index_vpsa(elementsGPU, sets);
    fprintf('\t Available Memory on GPU after INDEX creation:   \t %G MB \n', d.AvailableMemory/1e6);
    Ked = eStiff_vpsa(elementsGPU, nodesGPU, MP, sets);
    fprintf('\t Available Memory on GPU after KE creation:      \t %G MB \n', d.AvailableMemory/1e6);
    clear elementsGPU nodesGPU
    wait(d);
    fprintf('\t Available Memory on GPU after mesh deletion:    \t %G MB \n', d.AvailableMemory/1e6);
    
    if (ndiv > 1)
        %  New
        nent = sets.nel * sets.sz;
        m = nent / ndiv;
        n = ndiv;
        iKd = reshape(iKd, m, n);
        jKd = reshape(jKd, m, n);
        Ked = reshape(Ked, m, n);
        K = sparse(sets.tdofs, sets.tdofs);
        for i=1:ndiv
            K = K + AssemblyStiffMa(iKd(:,1), jKd(:,1), Ked(:,1), sets);  % New
            fprintf('\t Available Memory on GPU after ASSEMBLY:         \t %G MB \n', d.AvailableMemory/1e6);
            iKd(:,1) = []; jKd(:,1) = []; Ked(:,1) = [];
            fprintf('\t Available Memory on GPU after DELETE:           \t %G MB \n', d.AvailableMemory/1e6);
        end
        wait(d);
        
    else
        % Old
        K = AssemblyStiffMa(iKd, jKd, Ked, sets);
        wait(d);
        
    end
    
end

%% Clear GPU memory
% reset(gpuDevice);
