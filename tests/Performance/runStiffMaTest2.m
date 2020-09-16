function fullTable = runStiffMaTest2
% Runs the ALL assembly code by varying problem size and problem type on GPU.
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.
%
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  21/05/2020. Version: 1.0


%% Variables for performance tests
nel_all  = [90];        % Cases for mesh size. Limited by GPU memory
sf_all   = 0:3:180;                % Safety factor. Positive integer to add more partitions
prob_all = {'Vector'};      % Cases for problem type

sets.dTE = 'uint32';       	% Cases for "element" data type
sets.dTN = 'double';      	% Cases for "nodes" data type
sets.proc_type = 'GPU';         % Cases for processor type

%% Adding folders to the path
addpath('../Scalar/');
addpath('../Vector/');
addpath('../Common');
addpath('../Utils');
addpath(pwd);

%% Move to results folder
mkdir 'StiffMa2_PerfTestRst/';
cd 'StiffMa2_PerfTestRst/';

%% Platform details
MWver = ver;        % Version information for MathWorks products
platform = system_dependent('getos');
infoCPU = cpuinfo();
infoGPU = gpuDevice();
sys_info = evalc('configinfo'); % Write system information
if ismac
    sets.pf = 'MAC';    % Code to run on Mac platform
    lshw = [];
elseif isunix
    sets.pf = 'LNX';    % Code to run on Linux platform
    lshw = evalc('!lshw'); % List hardware details
elseif ispc
    sets.pf = 'WIN';    % Code to run on Windows platform
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
    
    % Loop through sf
    for i = 1:length(sf_all)
        sets.sf = sf_all(i);
        
        % Loop through problem type
        for pbl = 1:length(prob_all)
            sets.prob_type = prob_all{pbl};
            
            % Prepares the test
            sets.name = ['StiffMa2_',sets.pf,'_',sets.prob_type(1:3),'_sf',num2str(sets.sf),'_nel',num2str(sets.nel)];
            ndiv = WriteStiffMaPerfScript2(sets);
            fprintf("\n\nStarting the performance measurement with the following parameters:\n");
            fprintf("Number of finine elements: %dx%dx%d (%d)\n",sets.nel,sets.nel,sets.nel,sets.nel^3);
            fprintf("Date type for 'elements': '%s'\n",sets.dTE);
            fprintf("Date type for 'nodes': '%s'\n",sets.dTN);
            fprintf("Problem type: '%s'\n",sets.prob_type);
            fprintf("Processor type: '%s'\n\n",sets.proc_type);
            
            % Executes the performance test
            if ndiv > 50
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
                fullTable = vertcat(fullTable, perf_rst.sampleSummary);  %#ok Colects the statistics for all the test cases
            end
            save(sets.name,'perf_rst','fullTable','sets'); % Save partial results
            reset(gpuDevice);
            
        end
    end
end

%% Save total results
fname = ['StiffMa2_PerfTestRST_',sets.pf];
save(fname,'fullTable');
fprintf('\n\nA total of %i time experiments was executed!\n',it)
fprintf('Date and time at the beginning of the process: \t%s \n',datestr(t));
fprintf('Date and time at the end of the process: \t%s\n\n',datestr(now));
cd ..
