function K = StiffMa(Mesh, MP, dev, sets)
% STIFFMA Creates the global sparse stiffness matrix on the GPU by dividing
% [iK, jK, Ke] arrays if the device memory is not enough to process all data
% together.
% 
%   K = STIFFMA(Mesh, MP, dev, sets) returns the lower-triangle of a sparse
%   matrix K from finite element analysis of scalar/vector problems in a 
%   three-dimensional domain taking advantage of symmetry and GPU
%   computing, where the required inputs are:
%   - "Mesh.elements" is the connectivity matrix,
%   - "Mesh.nodes" is the nodal coordinates, 
%   - "MP.E" is the Young's modulus and
%   - "MP.nu" is the Poisson ratio are the material property for an isotropic
%   material in the vector case, while 
%   - "MP.c" (thermal conductivity) is needed for the scalar problem. The
%   struct "sets" must contain several similation parameters: 
%   - sets.prob_type defines the problem type 'Scalar' or 'Vector'
%   - sets.sf is the safety factor. Positive integer to add more partitions
%   - sets.dTE is the data precision of "Mesh.elements"
%   - sets.dTN is the data precision of "Mesh.nodes"
%   - sets.nel is the number of finite elements
%   - sets.nxe is the number of nodes per element
%   - sets.nnod is the number of nodes in the mesh 
%   - sets.dim is the space dimension
%   - sets.dxn is the number of DOFs per node. 1 for the scalar problem or 3 for the vector problem
%   - sets.edof is the umber of DOFs per element
%   - sets.tdofs is the number of total DOFs in the mesh
%   - sets.sz  is the number of symmetry entries
%   - sets.tbs is the Thread Block Size
%   - sets.numSMs is the number of multiprocessors on the device
%   - sets.WarpSize is the GPU warp size

%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.
%
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  21 May 2020. Version: 1.0
%   Modified: 15 June 2020. Version: 1.1


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

