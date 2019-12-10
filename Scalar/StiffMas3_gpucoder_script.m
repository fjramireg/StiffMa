% STIFFMAS3_GPUCODER_SCRIPT   Generate MEX-function StiffMas3_mex from
%  StiffMas3.
% 
% Script generated from project 'StiffMas3.prj' on 09-Dec-2019.
% 
% See also CODER, CODER.CONFIG, CODER.TYPEOF, CODEGEN.

%% Create configuration object of class 'coder.MexCodeConfig'.
cfg = coder.gpuConfig('mex');
cfg.InitFltsAndDblsToZero = false;
cfg.GenerateReport = true;
cfg.ReportPotentialDifferences = false;
cfg.GpuConfig.CompilerFlags = '--fmad=false'; % instructs the compiler to disable Floating-Point Multiply-Add (FMAD) optimization. This option is set to prevent numerical mismatch in the generated code because of architectural dLfferences in the CPU and the GPU. 
cfg.GpuConfig.CustomComputeCapability = '-arch=sm_50'; % You can specify a real architecture

%% Define argument types for entry-point 'StiffMas3'.
ARGS = cell(1,1);
ARGS{1} = cell(3,1);
ARGS{1}{1} = coder.typeof(uint32(0),[Inf  8],[1 0]);
ARGS{1}{2} = coder.typeof(0,[Inf  3],[1 0]);
ARGS{1}{3} = coder.typeof(0);

%% Invoke MATLAB Coder.
codegen -config cfg StiffMas3 -args ARGS{1}
% The -report option instructs codegen to generate a code generation report that you can use to debug your MATLAB code.
% The -args option instructs codegen to compile the file StiffMas3.m by using the class, size, and complexity of the input parameters.
% The -config option instructs codegen to use the specified configuration object for code generation.

