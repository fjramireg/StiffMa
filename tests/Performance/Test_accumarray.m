nel = 60;
density = 0.001;
nnel = uint32(nel^3);
nnod = (nel+1)^3;

%% CPU 
R = sprand(nnod, nnod, density);

%% GPU
% d = gpuDevice;
% fprintf('\t Available Memory on GPU before start:       \t %G MB \n', d.AvailableMemory/1e6);
% 
% ik = 1:36*nnel;
% jk = ik;
% ke = rand(1,36*nnel);
% fprintf('\t Available Memory on GPU before accumarray:   \t %G MB \n', d.AvailableMemory/1e6);
% K = accumarray([ik',jk'], ke', [], [], [], 1);
% fprintf('\t Available Memory on GPU after accumarray:   \t %G MB \n', d.AvailableMemory/1e6);
