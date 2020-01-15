% This script is created from coder.checkGpuInstallApp on 14-Jan-2020 17:50:51 
 

% Selecting Gpu 
gpuDevice(1);

% CUDA Installation Path 
setenv('LD_LIBRARY_PATH', ['/usr/local/cuda-10.2/lib64' pathsep getenv('LD_LIBRARY_PATH')]);
setenv('PATH', ['/usr/local/cuda-10.2/bin' pathsep getenv('PATH')]);

% cuDNN Installation Path 
setenv('NVIDIA_CUDNN','/usr/local/cuda-10.2');

% TensorRT Installation Path 
setenv('NVIDIA_TENSORRT','/home/francisco/Documents/TensorRT7.0/TensorRT-7.0.0.11');

% NVTX Library Path 
setenv('LD_LIBRARY_PATH', ['/usr/local/cuda-10.2/lib64' pathsep getenv('LD_LIBRARY_PATH')]);