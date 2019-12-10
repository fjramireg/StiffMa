% This script is created from coder.checkGpuInstallApp on 08-Dec-2019 07:51:04 
 

% Selecting Gpu 
gpuDevice(1);

% CUDA Installation Path 
setenv('LD_LIBRARY_PATH', ['/usr/local/cuda-10.1/lib64' pathsep getenv('LD_LIBRARY_PATH')]);
setenv('PATH', ['/usr/local/cuda-10.1/bin' pathsep getenv('PATH')]);