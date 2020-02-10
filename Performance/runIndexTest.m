function runIndexTest
% Runs the INDEX code on the GPU by varying problem size and data precision type
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.
%
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  03/02/2020. Version: 1.4

% Adding folders to the path
addpath('../Scalar/');
addpath('../Vector/');
addpath('../Common');
addpath('../Utils');
addpath(pwd);

% Platform details
MWver = ver;        % Version information for MathWorks products
Matlab_v = version; % Version number for MATLAB and libraries
platform = system_dependent('getos');
infoCPU = cpuinfo();
infoGPU = gpuDevice();
if ismac
    sets.pf = 'MAC';    % Code to run on Mac platform
elseif isunix
    sets.pf = 'LNX';    % Code to run on Linux platform
    system('lshw');% List hardware details
elseif ispc
    sets.pf = 'WIN';    % Code to run on Windows platform
else
    error('Platform not supported');
end

% Variables for performance tests
nel_all  = [10,20,40,80,160,320,640,1000]; 	% Cases for mesh size
dTEall = {'uint32','uint64'};           % Cases for "element" data type
dTNall = {'single'};                   	% Cases for "nodes" data type

% Move to results folder
mkdir 'IndexPerfTestRst/';
cd 'IndexPerfTestRst/';

% Runs all tests
for k = 1:length(nel_all)
    sets.nel = nel_all(k);
    
    for i = 1:length(dTEall)
        sets.dTE = dTEall{i};
        
        for j = 1:length(dTNall)
            sets.dTN = dTNall{j};
            
            sets.name = ['IndexTest_',sets.pf,'_N',sets.dTN,'_E',sets.dTE,'_nel',num2str(sets.nel)];
            WriteIndexPerfScript2(sets);
            fprintf("\n\nStarting the performance measurement with the following parameters:\n");
            fprintf("Number of finine elements: %dx%dx%d (%d)\n",sets.nel,sets.nel,sets.nel,sets.nel^3);
            fprintf("Date type for 'elements': '%s'\n",sets.dTE);
            fprintf("Date type for 'nodes': '%s'\n",sets.dTN);
            perf_rst = runperf(sets.name); disp(perf_rst);
            save(sets.name,'perf_rst','sets','Matlab_v','MWver','platform','infoCPU','infoGPU');
            reset(gpuDevice);
            % FIXME: Error occurred in IndexTest_LNX_Nsingle_Euint32_nel20/Index_GPU_Vector_Symmetry and it did not run to completion.
            %     ---------
            %     Error ID:
            %     ---------
            %     'parallel:gpu:device:UnknownCUDAError'
            %     --------------
            %     Error Details:
            %     --------------
            %     Error using parallel.gpu.CUDADevice/wait
            %     An unexpected error occurred during CUDA execution. The CUDA error was:
            %     an illegal memory access was encountered
            %
            %     Error in IndexTest_LNX_Nsingle_Euint32_nel20 (line 34)
            
        end
    end
end
