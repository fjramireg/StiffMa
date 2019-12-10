% STIFFMAS_GPUCODER_SCRIPT   Generate MEX-function StiffMas_mex from StiffMas.
% 
% Script generated from project 'StiffMas.prj' on 08-Dec-2019.
% 
% See also CODER, CODER.CONFIG, CODER.TYPEOF, CODEGEN.

%% Create configuration object of class 'coder.MexCodeConfig'.
cfg = coder.gpuConfig('mex');
cfg.GpuConfig.CompilerFlags = '--fmad=false'; % instructs the compiler to disable Floating-Point Multiply-Add (FMAD) optimization. This option is set to prevent numerical mismatch in the generated code because of architectural dLfferences in the CPU and the GPU. 
cfg.GpuConfig.CustomComputeCapability = '-arch=sm_50'; % You can specify a real architecture
cfg.InitFltsAndDblsToZero = false;
cfg.MATLABSourceComments = true;
cfg.GenerateReport = true;
cfg.LaunchReport = true;
cfg.ReportPotentialDifferences = false;

%% Invoke MATLAB Coder.
codegen -config cfg StiffMas

