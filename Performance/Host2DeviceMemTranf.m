function [e_d, n_d] = Host2DeviceMemTranf(e,n)
% Transfer two arrays from the host (CPU) to the device (GPU)

e_d = gpuArray(e');
n_d = gpuArray(n');
