function info = gpuinfo()
%GPUINFO  get basic details about the GPU being used
%
%   GPUINFO() gets hold of some basic info about the GPU device being used.
%
%   See also: CPUINFO
% 
%   For more information, see the <a href="matlab:
%   web('https://www.mathworks.com/matlabcentral/fileexchange/34080-gpubench')"
%   >Matlab Central</a> web site.

%   Copyright 2011-2014 The MathWorks, Inc.

if parallel.gpu.GPUDevice.isAvailable()
    gpu = gpuDevice();
    info = struct( ...
        'Name', gpu.Name, ...
        'Clock', sprintf( '%u MHz', gpu.ClockRateKHz/1e3 ), ...
        'NumProcessors', gpu.MultiprocessorCount, ...
        'ComputeCapability', gpu.ComputeCapability, ...
        'TotalMemory', sprintf( '%1.2f GB', gpu.TotalMemory/2^30 ), ...
        'CUDAVersion', gpu.DriverVersion, ...
        'DriverVersion', parallel.internal.gpu.CUDADriverVersion );
else
    % No GPU, so create an empty structure
    info = struct( ...
        'Name', '<no GPU available>', ...
        'Clock', '', ...
        'NumProcessors', 0, ...
        'ComputeCapability', '', ...
        'TotalMemory', 0, ...
        'CUDAVersion', '', ...
        'DriverVersion', '' );
end

