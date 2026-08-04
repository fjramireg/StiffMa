function fullTable = runIndexTest2026
% Runs the INDEX code by varying problem size, data precision type, problem type
% and processor type.
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.

%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Institución Universitaria Pascual Bravo, Medellin-Colombia
%       Updated: July 24, 2026.
%       Created:  13/02/2020. Version: 1.4

%% Max theoretical nel
nxSca32 = computeNelmaxGPU(4, 36, 2);  % uint32_scalar
nxVec32 = computeNelmaxGPU(4, 300, 2); % uint32_vector
nxSca64 = computeNelmaxGPU(8, 36, 2);  % uint64_scalar
nxVec64 = computeNelmaxGPU(8, 300, 2); % uint64_vector

fprintf('\n\n The maximum theoretical number of finite elements is:\n')
fprintf('       uint32_scalar: %i\n',nxSca32)
fprintf('       uint32_vector: %i\n',nxVec32)
fprintf('       uint64_scalar: %i\n',nxSca64)
fprintf('       uint64_vector: %i\n',nxVec64)

%% Variables for performance tests
% nel_all = [5 10];        % Toy
nel_all0 = [10 20 40 80 160 320];% Cases for mesh size.
% % nel_all1 = [nxSca32-5:nxSca32+5, nxVec32-5:nxVec32+5, nxSca64-5:nxSca64+5, nxVec64-5:nxVec64+5]; % Limited by GPU memory (OOM)
nel_all1 = [nxSca32, nxVec32, nxSca64, nxVec64]; % Limited by GPU memory (OOM)
nel_all = sort(unique([nel_all0, nel_all1]));
dTEall  = {'uint32','uint64'};          % Cases for "element" data type
dTNall  = {'single'};                   % Cases for "nodes" data type. Do not matter for this test
prob_all= {'Scalar','Vector'};          % Cases for problem type
proc_all= {'CPU','GPU'};                % Cases for processor type

%% Save results in this folder
old = pwd;
cd('../../')
addpath(genpath(pwd));
cd(old);
folder = 'PerfTest_2026';
if isfolder(folder)
    cd(folder);
else
    mkdir(folder);
    cd(folder);
end

%% Platform details
MWver = ver;            %#ok % Version information for MathWorks products
platform = system_dependent('getos'); %#ok<NASGU> % OS info
infoCPU = cpuinfo;      %#ok % CPU info
infoGPU = gpuDevice;    %#ok % CPU info
sys_info = evalc('configinfo'); %#ok % Write system information
if ismac
    sets.pf = 'MAC';    % Code to run on Mac platform
    lshw = [];          %#ok<NASGU>
elseif isunix
    sets.pf = 'LNX';    % Code to run on Linux platform
    lshw = evalc('!lshw'); %#ok % List hardware details
elseif ispc
    sets.pf = 'WIN';    % Code to run on Windows platform
    lshw = [];          %#ok<NASGU>
else
    error('Platform not supported');
end

%% Runs all INDEX tests
t0 = datetime('now');                   % Current date and time at starting the process
it = 0;                                 % Counter
import matlab.perftest.TimeExperiment   % To customize the time experiment

% Loop through mesh size
for k = 1:length(nel_all)
    sets.nel = nel_all(k);

    % Loop through element connectivity precision
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
                    sets.name = 'IndexTest';
                    Filename = [sets.name,'.m'];
                    WriteIndexPerfScript2026(sets);
                    type(Filename);

                     % Executes the performance test
                    if ( strcmp(sets.proc_type,'CPU') && sets.nel > 40  )   % Takes long time to run the full test. So, only 1 execution is taken
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
                        reset(gpuDevice);
                    end

                    % Partial results
                    it = it + 1;
                    if it == 1
                        fullTable = vertcat(perf_rst.sampleSummary);
                    else
                        fullTable = vertcat(fullTable, perf_rst.sampleSummary);  % Collects the statistics for all the test cases
                    end

                end
            end
        end
    end
end
delete(Filename);
t1 = datetime('now');                   % Current date and time at the END of the process

%% Save total results
fname = ['IndexPerfTestOut_',sets.pf,'2026.mat'];
save(fname);
fprintf('\n\nA total of %i time experiments was executed!\n',it)
fprintf('Date and time at the beginning of the process: \t%s \n',t0);
fprintf('Date and time at the end of the process: \t%s\n',t1);
fprintf('Elapsed time : \t%s\n\n',t1-t0);
