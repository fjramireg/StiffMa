function fullTable = runEStiffTest
% Runs the ESTIFF code by varying problem size, data precision type, problem type
% and processor type.
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.
%
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  14/02/2020. Version: 1.4


%% Variables for performance tests
nel_all = [10,20,40,80,160,320];    % Cases for mesh size
dTEall = {'uint32'};            % Cases for "element" data type
dTNall = {'single','double'};   % Cases for "nodes" data type
prob_all = {'Scalar','Vector'};	% Cases for problem type
proc_all = {'GPU'};       % Cases for processor type

%% Adding folders to the path
addpath('../Scalar/');
addpath('../Vector/');
addpath('../Common');
addpath('../Utils');
addpath(pwd);

%% Save results in this folder
mkdir 's_EStiffPerfTestRst/';
cd 's_EStiffPerfTestRst/';

%% Platform details
MWver = ver;                    % Version information for MathWorks products
platform = system_dependent('getos');
infoCPU = cpuinfo();
infoGPU = gpuDevice();
sys_info = evalc('configinfo');	% Write system information
if ismac
    sets.pf = 'MAC';            % Code to run on Mac platform
    lshw = [];
elseif isunix
    sets.pf = 'LNX';            % Code to run on Linux platform
    lshw = evalc('!lshw');      % List hardware details
elseif ispc
    sets.pf = 'WIN';            % Code to run on Windows platform
    lshw = [];
else
    error('Platform not supported');
end

%% Runs all tests
t = now;                                % Current date and time at starting the process
it = 0;                                 % Couter
import matlab.perftest.TimeExperiment   % To customize the time experiment

% Loop through mesh size
for k = 1:length(nel_all)
    sets.nel = nel_all(k);
    
    % Loop through element conectivity precision
    for i = 1:length(dTEall)
        sets.dTE = dTEall{i};
        
        % Loop through nodal coordinates precision
        for j = 1:length(dTNall)
            sets.dTN = dTNall{j};
            
            % Loop through problem type
            for pbl = 1:length(prob_all)
                sets.prob_type = prob_all{pbl};
                
                % Loop through processor type
                for proc = 1:length(proc_all)
                    sets.proc_type = proc_all{proc};
                    
                    % Prepares the test
                    sets.name = ['ETest',sets.pf,'_',sets.proc_type(1),sets.prob_type(1),...
                        sets.dTN(1),sets.dTE(end-1:end),'_',num2str(sets.nel)];
                    WriteEStiffPerfScript(sets);
                    fprintf("\n\nStarting the performance measurement with the following parameters:\n");
                    fprintf("Number of finine elements: %dx%dx%d (%d)\n",sets.nel,sets.nel,sets.nel,sets.nel^3);
                    fprintf("Date type for 'elements': '%s'\n",sets.dTE);
                    fprintf("Date type for 'nodes': '%s'\n",sets.dTN);
                    fprintf("Problem type: '%s'\n",sets.prob_type);
                    fprintf("Processor type: '%s'\n\n",sets.proc_type);
                    
                    % Executes the performance test
                    if ( strcmp(sets.proc_type,'CPU') && sets.nel > 20  )   % Takes long time to run the full test. So, only 1 execution is taken
                        numSamples = 1;                                         % Number of sample measurements to collect, specified as a positive integer.
                        numWarmups = 0;                                         % Number of warm-up measurements, specified as a nonnegative integer.
                        suite = testsuite(sets.name);                           % Construct an explicit test suite
                        experiment = TimeExperiment.withFixedSampleSize(numSamples,'NumWarmups',numWarmups);% Construct time experiment with fixed number of measurements
                        perf_rst = run(experiment,suite);
                        disp(perf_rst);
                        
                    else                % Default experiment setup
                        % Number of warm-up measurements: 4
                        % Minimum number of samples: 4
                        % Maximum number of samples collected in the event other statistical objectives are not met: 256
                        % Objective relative margin of error for samples: 0.05 (5%)
                        % Confidence level for samples to be within relative margin of error: 0.95 (95%)
                        perf_rst = runperf(sets.name);
                        disp(perf_rst);
                        
                    end
                    
                    % Partial results
                    it = it + 1;
                    if it == 1
                        fullTable = vertcat(perf_rst.sampleSummary);
                    else
                        fullTable = vertcat(fullTable, perf_rst.sampleSummary);  % Colects the statistics for all the test cases
                    end
                    save(sets.name,'perf_rst','fullTable',...
                        'sets','MWver','platform','infoCPU','infoGPU','sys_info','lshw'); % Save partial results
                    reset(gpuDevice);
                    
                end
            end
        end
    end
end

%% Save total results
fname = ['EStiffPerfTest_',sets.pf];
save(fname,'fullTable');
fprintf('\n\n A total of %i time experiments was executed!\n',it)
fprintf('Date and time at the beginning of the process: \t%s \n',datestr(t));
fprintf('Date and time at the end of the process: \t%s\n\n',datestr(now));
