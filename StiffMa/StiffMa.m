function K = StiffMa(Mesh, MP, dev, sets)
% Script to test the StiffMa code by dividing [iK, jK, Ke] arrays

%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.
%
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  21/05/2020. Version: 1.0


%% Determination of number of chunks based on current GPU memory

% Element data type used for storing connectivity array (Mesh.elements)
d_et  = zeros(1,1,sets.dTE); %#ok
d_et1 = whos('d_et');
szInd = d_et1.bytes;

% Nodal data type used for storing nodal coordinates array (Mesh.nodes)
d_nt  = zeros(1,1,sets.dTN); %#ok
d_nt1 = whos('d_nt');
szNNZ = d_nt1.bytes;

% Memory requirements
Mmesh  = szInd*numel(Mesh.elements) + szNNZ*numel(Mesh.nodes);	% Memory required to store the mesh
Mtrip  = (2*szInd + szNNZ)*sets.sz*sets.nel;                    % Memory required to store triplet format iK, jK, Ke
Maccum = 3*Mtrip;                                               % Memory required for accumarray
Mcsc   = 0.5*Mtrip;                                             % Memory required for output (sparse metrix in CSC format)
Mtotal = Mmesh + Mtrip + Maccum + Mcsc;                        	% Total required memory

% Chunks determination
ndiv = ceil(Mtotal/dev.AvailableMemory);                          % Number of divisions.
ndiv = ndiv + sets.sf*(ndiv>1);                                      % Sums safety factor only if ndiv > 1
while mod(sets.nel,ndiv) ~= 0                                   % Guarantee that sets.nel is divisible by ndiv
    ndiv = ndiv + 1;
end

% Determination of memory required for global K
Mtotal_c = Mmesh + Mtrip*(4/ndiv + 1/2);
if Mtotal_c > dev.AvailableMemory
    reset(dev);
    error('Error!... GPU memory is not enough to allocate the global sparse stiffness matrix. Try to increase sf.');
else
    fprintf('The global stiffness matrix will be computed with %d chunk(s).\n',ndiv);
end

%% Transfer memory: host to device
fprintf('Available memory on GPU before computations begin: %4.2f MB \n', dev.AvailableMemory/1e6);
elementsGPU = gpuArray(Mesh.elements');
nodesGPU = gpuArray(Mesh.nodes');

%% Stiffness matrix generation

% Scalar
if strcmp(sets.prob_type,'Scalar')
    % Partitioning
    if (ndiv > 1)
        m = sets.nel / ndiv;
        sets.nel = m;
        K = sparse(sets.tdofs, sets.tdofs);
        for i=1:ndiv
            fprintf('\t Processing Chunk %d of %d...\n',i,ndiv);
            ini = 1 + m*(i-1);
            fin = ini + m - 1;
            [iKd, jKd] = Index_spsa(elementsGPU(:, ini:fin), sets);
            Ked = eStiff_spsa(elementsGPU(:, ini:fin), nodesGPU, MP.c, sets);
            wait(dev);
            K = K + AssemblyStiffMa(iKd, jKd, Ked, sets);
        end
        clear elementsGPU nodesGPU iKd jKd Ked
        % Without partitioning
    else
        [iKd, jKd] = Index_spsa(elementsGPU, sets);
        Ked = eStiff_spsa(elementsGPU, nodesGPU, MP.c, sets);
        wait(dev);
        clear elementsGPU nodesGPU
        K = AssemblyStiffMa(iKd, jKd, Ked, sets);
        clear iKd jKd Ked
    end
    
    % Vector
elseif strcmp(sets.prob_type,'Vector')   
    % Partitioning
    if (ndiv > 1)
        m = sets.nel / ndiv;
        sets.nel = m;
        K = sparse(sets.tdofs, sets.tdofs);
        for i=1:ndiv
            fprintf('\t Processing Chunk %d of %d...\n',i,ndiv);
            ini = 1 + m*(i-1);
            fin = ini + m - 1;
            [iKd, jKd] = Index_vpsa(elementsGPU(:, ini:fin), sets);
            Ked = eStiff_vpsa(elementsGPU(:, ini:fin), nodesGPU, MP, sets);
            wait(dev);
            K = K + AssemblyStiffMa(iKd, jKd, Ked, sets);  % New
        end
        clear elementsGPU nodesGPU iKd jKd Ked
    % Without partitioning    
    else            
        [iKd, jKd] = Index_vpsa(elementsGPU, sets);
        Ked = eStiff_vpsa(elementsGPU, nodesGPU, MP, sets);
        wait(dev);
        clear elementsGPU nodesGPU
        K = AssemblyStiffMa(iKd, jKd, Ked, sets);
        clear iKd jKd Ked        
    end
    
end

fprintf('Available memory on GPU after computations: %4.2f MB \n', dev.AvailableMemory/1e6);

