% Script to test the StiffMa code by dividing [iK, jK, Ke] arrays

nel = 180;

sf = 1; % Safety factor. Positive integer to add more partitions
Problem = 'Scalar';     % 'Scalar' or 'Vector'
sets.dTE = 'uint32';
sets.dTN = 'double';

%% GPU setup
d = gpuDevice;
sets.tbs = d.MaxThreadsPerBlock;
sets.numSMs   = d.MultiprocessorCount;
sets.WarpSize = d.SIMDWidth;

%% Adding folders to the path
addpath('../Scalar/');
addpath('../Vector/');
addpath('../Common');
addpath('../Utils');
addpath(pwd);

%% Mesh creation
[Mesh.elements, Mesh.nodes] = CreateMesh2(nel, nel, nel, sets.dTE, sets.dTN);
[sets.nel, sets.enod] = size(Mesh.elements);        % Number of elements in the mesh & Number of nodes per element
[sets.tnod, sets.dim] = size(Mesh.nodes);           % Number of nodes in the mesh & Space dimension

if strcmp(Problem,'Scalar')
    sets.ndof = 1;                                  % Number of DOFs per node
elseif strcmp(Problem,'Vector')
    sets.ndof = 3;                                  % Number of DOFs per node
else
    error('Problem not defined!');
end
sets.edof = sets.ndof * sets.enod;                  % Number of DOFs per element
sets.sz = (sets.edof * (sets.edof + 1) )/2;         % Number of NNZ values for each Ke
sets.tdofs = sets.tnod * sets.ndof;                 % Number of total DOFs in the mesh

%% Partitioning -  Determination of number of divisions based on GPU memory

% Element data type for storing connectivity array (Mesh.elements)
d_et = zeros(1,1,sets.dTE);
d_et1 = whos('d_et');
szInd = d_et1.bytes;

% Nodal data type for storing nodal coordinates array (Mesh.nodes)
d_nt = zeros(1,1,sets.dTN);
d_nt1 = whos('d_nt');
szNNZ = d_nt1.bytes;

% Memory requirements
mmr = szInd*numel(Mesh.elements) + szNNZ*numel(Mesh.nodes);	% Memory required to store the mesh
trim = (2*szInd + szNNZ)*sets.sz*sets.nel;                  % Memory required to store triplet format iK, jK, Ke
accm = 3*trim;                                              % Memory required for accumarray
cscm = 0.5*trim;                                            % Memory required for output (sparse metrix in CSC format)
trm = mmr + trim + accm + cscm;                             % Total required memory

% Chunks determination
ndiv = ceil(trm/d.AvailableMemory);                         % Number of divisions.
ndiv = ndiv + sf*(ndiv>1);                                  % Sums safety factor only if ndiv > 1
while mod(sets.nel,ndiv) ~= 0                               % Guarantee that sets.nel is divisible by ndiv
    ndiv = ndiv + 1;
end

% Determination of memory required for global K
mr4K = mmr + 4*trim/ndiv + cscm;
if mr4K > d.AvailableMemory
    reset(gpuDevice);
    error('Error!... GPU memory is not enough to allocate the global sparse stiffness matrix. Try to increase sf.');
else
    fprintf('The global stiffness matrix will be computed with %d chunk(s).\n',ndiv);
end

%% Transfer memory: host to device
fprintf('\t Available Memory on GPU before start:           \t %4.2f MB \n', d.AvailableMemory/1e6);
elementsGPU = gpuArray(Mesh.elements');
nodesGPU = gpuArray(Mesh.nodes');
fprintf('\t Available Memory on GPU after mesh transfer:    \t %4.2f MB \n', d.AvailableMemory/1e6);

%% Stiffness matrix generation

% Scalar
if strcmp(Problem,'Scalar')
    c = 384.1;
    
    if (ndiv > 1)   % Partitioning
        m = sets.nel / ndiv;
        sets.nel = m;
        K = sparse(sets.tdofs, sets.tdofs);
        for i=1:ndiv
            fprintf('Processing Chunk %d of %d...\n',i,ndiv);
            ini = 1 + m*(i-1);
            fin = ini + m - 1;
            [iKd, jKd] = Index_spsa(elementsGPU(:, ini:fin), sets);
            fprintf('\t Available Memory on GPU after INDEX creation:   \t %4.2f MB \n', d.AvailableMemory/1e6);
            Ked = eStiff_spsa(elementsGPU(:, ini:fin), nodesGPU, c, sets);
            wait(d);
            fprintf('\t Available Memory on GPU after KE creation:      \t %4.2f MB \n', d.AvailableMemory/1e6);
            K = K + AssemblyStiffMa(iKd, jKd, Ked, sets);  % New
            fprintf('\t Available Memory on GPU after ASSEMBLY:         \t %4.2f MB \n', d.AvailableMemory/1e6);
        end
        clear elementsGPU nodesGPU iKd jKd Ked
        fprintf('\t Available Memory on GPU after DELETE:           \t %4.2f MB \n', d.AvailableMemory/1e6);
        
    else            % Without partitioning
        [iKd, jKd] = Index_spsa(elementsGPU, sets);
        fprintf('\t Available Memory on GPU after INDEX creation:   \t %4.2f MB \n', d.AvailableMemory/1e6);
        Ked = eStiff_spsa(elementsGPU, nodesGPU, c, sets);
        fprintf('\t Available Memory on GPU after KE creation:      \t %4.2f MB \n', d.AvailableMemory/1e6);
        wait(d);
        clear elementsGPU nodesGPU
        K = AssemblyStiffMa(iKd, jKd, Ked, sets);
        fprintf('\t Available Memory on GPU after ASSEMBLY:         \t %4.2f MB \n', d.AvailableMemory/1e6);
        clear iKd jKd Ked
        
    end
    
    
    %% Vector
elseif strcmp(Problem,'Vector')
    MP.E = 200e9;
    MP.nu = 0.3;
    
    if (ndiv > 1)   % Partitioning
        m = sets.nel / ndiv;
        sets.nel = m;
        K = sparse(sets.tdofs, sets.tdofs);
        for i=1:ndiv
            fprintf('Processing Chunk %d of %d...\n',i,ndiv);
            ini = 1 + m*(i-1);
            fin = ini + m - 1;
            [iKd, jKd] = Index_vpsa(elementsGPU(:, ini:fin), sets);
            fprintf('\t Available Memory on GPU after INDEX creation:   \t %4.2f MB \n', d.AvailableMemory/1e6);
            Ked = eStiff_vpsa(elementsGPU(:, ini:fin), nodesGPU, MP, sets);
            wait(d);
            fprintf('\t Available Memory on GPU after KE creation:      \t %4.2f MB \n', d.AvailableMemory/1e6);
            K = K + AssemblyStiffMa(iKd, jKd, Ked, sets);  % New
            fprintf('\t Available Memory on GPU after ASSEMBLY:         \t %4.2f MB \n', d.AvailableMemory/1e6);
        end
        clear elementsGPU nodesGPU iKd jKd Ked
        
    else            % Without partitioning
        [iKd, jKd] = Index_vpsa(elementsGPU, sets);
        fprintf('\t Available Memory on GPU after INDEX creation:   \t %4.2f MB \n', d.AvailableMemory/1e6);
        Ked = eStiff_vpsa(elementsGPU, nodesGPU, MP, sets);
        fprintf('\t Available Memory on GPU after KE creation:      \t %4.2f MB \n', d.AvailableMemory/1e6);
        wait(d);
        clear elementsGPU nodesGPU
        fprintf('\t Available Memory on GPU after mesh deletion:    \t %4.2f MB \n', d.AvailableMemory/1e6);
        K = AssemblyStiffMa(iKd, jKd, Ked, sets);
        clear iKd jKd Ked
        
    end
    
end

%% Clear GPU memory
reset(gpuDevice);
