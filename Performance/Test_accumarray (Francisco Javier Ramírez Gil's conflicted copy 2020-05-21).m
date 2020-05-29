nel = 160;
density = 0.00001;
nnel = uint32(nel^3);
nnod = (nel+1)^3;

%% CPU 
R = sprand(nnod, nnod, density);
[row,col,v] = find(R);

%% GPU
d = gpuDevice;
fprintf('\t Available Memory on GPU before start:       \t %G MB \n', d.AvailableMemory/1e6);
ik = gpuArray(row);
jk = gpuArray(col);
ke = gpuArray(v);
fprintf('\t Available Memory on GPU before accumarray:   \t %G MB \n', d.AvailableMemory/1e6);
K = accumarray([ik,jk], ke, [], [], [], 1);
fprintf('\t Available Memory on GPU after accumarray:   \t %G MB \n', d.AvailableMemory/1e6);

% Hi there,
% I'm currently trying to create sparse matrices on GPU by using the accumarray function as K = accumarray([ik,jk], ke, [], [], [], 1). However, my GPU memory (4 GB) is not enough to work with. Therefore, I'd like to know how much temporal memory requires acumarray based on its inputs ik, jk & ke. This inputs comes finite element meshes, in which ik and jk are uint32 data type, while ke is double.
    
