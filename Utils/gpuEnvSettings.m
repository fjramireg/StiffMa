% This script is created for CUDA code compilation on 08-Dec-2019 07:51:04 
 
% Selecting Gpu 
gpu = gpuDevice(1)

% CUDA Installation Path 
setenv('LD_LIBRARY_PATH', ['/usr/local/cuda-10.2/lib64' pathsep getenv('LD_LIBRARY_PATH')]);
setenv('PATH', ['/usr/local/cuda-10.2/bin' pathsep getenv('PATH')]); %setenv('PATH',[getenv('PATH') ':/usr/local/cuda-10.2/bin']);
setenv('MW_NVCC_PATH','/usr/local/cuda-10.2/bin');

