
%% Prerequisite Products
% To use GPU Coder for CUDA C/C++ code generation, you must install the following products:
% • MATLAB (required).
% • MATLAB Coder (required).
% • Parallel Computing Toolbox (required).
% • Deep Learning Toolbox (required for deep learning).
% • GPU Coder Interface for Deep Learning Libraries (required for deep learning).
% • Image Processing Toolbox (recommended).
% • Computer Vision Toolbox™ (recommended).
% • Embedded Coder (recommended).
% • Simulink® (recommended).
% check which MathWorks products are installed
ver MATLAB
v = ver;
for k = 1:length(v)
    pN = v(k).Name;
    % Required packages
    if strcmp(pN,'GPU Coder') || strcmp(pN,'MATLAB Coder') ...
            || strcmp(pN,'Parallel Computing Toolbox')
        fprintf('%50s       Version %s \n', v(k).Name, v(k).Version);
    end
    % Recommended packages
    if strcmp(pN,'Deep Learning Toolbox') || strcmp(pN,'Image Processing Toolbox') ...
            || strcmp(pN,'Computer Vision Toolbox') || strcmp(pN,'Embedded Coder') ...
            || strcmp(pN,'Simulink')
        fprintf('%50s       Version %s \n', v(k).Name, v(k).Version);
    end
end

%% Third-party Products
d = gpuDevice(1);

% NVIDIA GPU enabled for CUDA with compute capability 3.2 or higher
if str2double(d.ComputeCapability) >= 3.2
    fprintf('NVIDIA GPU enable for CUDA computation!\n');
else
    fprintf('NVIDIA GPU not enable for CUDA computation\n!');
end

% Verify CUDA toolkit and driver. GPU Coder has been tested with CUDA toolkit v10.1 
if (d.DriverVersion >= 10.1 && d.ToolkitVersion >= 10.1)
    fprintf('Correct CUDA toolkit and driver version! \nDriverVersion: %f. \nToolkitVersion: %f\n',d.DriverVersion, d.ToolkitVersion); 
else
    fprintf('CUDA toolkit and driver version should be updated!\n');
end

% C/C++ Compiler. Linux: GCC C/C++ compiler 6.3.x
system('gcc --version');


%% Verify Setup: To verify that your development computer has all the tools and 
% configuration needed for GPU code generation
gpuEnvObj = coder.gpuEnvConfig('host');
gpuEnvObj.BasicCodegen = 1;
gpuEnvObj.BasicCodeexec = 1;
gpuEnvObj.DeepLibTarget = 'cudnn';
gpuEnvObj.DeepCodeexec = 1;
gpuEnvObj.DeepCodegen = 1;
gpuEnvObj.Profiling = 1;
results = coder.checkGpuInstall(gpuEnvObj);
