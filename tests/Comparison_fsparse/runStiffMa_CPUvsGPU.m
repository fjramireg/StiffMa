function fullTable = runStiffMa_CPUvsGPU
% Runs ALL code by varying problem size on vector problem for comparison: CPU vs GPU.
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.
%
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  June 22, 2020. Version: 1.0
%

%% Variables for performance tests
nel_all  = [10:10:100];        %#ok Cases for mesh size. Limited by GPU memory
sets.sf  = 1;                   % Safety factor. Positive integer to add more partitions
prob_all = {'Vector'};          % Cases for problem type
sets.dTE = 'uint32';            % Cases for "element" data type
sets.dTN = 'double';            % Cases for "nodes" data type
proc_type = {'CPU','GPU'};	% Cases for processor type

%% Adding folders to the path
addpath(pwd);
addpath(genpath('../../stenglib'));	% fparse
addpath(genpath('../../tbx'));      % StiffMa

%% Move to results folder
mkdir 'StiffMa_CPUvsGPU_R2020a';
cd 'StiffMa_CPUvsGPU_R2020a';

%% Runs all tests
t = now;                                % Current date and time at starting the process
it = 0;                                 % Couter
import matlab.perftest.TimeExperiment   % To customize the time experiment

% Loop through mesh size
for k = 1:length(nel_all)
    sets.nel = nel_all(k);
    
    % Loop through PROCESSOR
    for i = 1:length(proc_type)
        sets.proc_type = proc_type{i};
        
        % Loop through problem type
        for pbl = 1:length(prob_all)
            sets.prob_type = prob_all{pbl};
            
            % Prepares the test
            sets.name = ['Testing_',sets.proc_type];
            WriteStiffMaPerfScriptCPUvsGPU(sets);
            fprintf("\n\nStarting the performance measurement with the following parameters:\n");
            fprintf("Number of finine elements: %dx%dx%d (%d)\n",sets.nel,sets.nel,sets.nel,sets.nel^3);
            fprintf("Date type for 'elements': '%s'\n",sets.dTE);
            fprintf("Date type for 'nodes': '%s'\n",sets.dTN);
            fprintf("Problem type: '%s'\n",sets.prob_type);
            fprintf("Processor type: '%s'\n\n",sets.proc_type);
            
            % Executes the performance test
            perf_rst = runperf(sets.name);
            disp(perf_rst);
            
            % Partial results
            it = it + 1;
            if it == 1
                fullTable = vertcat(perf_rst.sampleSummary);
            else
                fullTable = vertcat(fullTable, perf_rst.sampleSummary);  %#ok Colects the statistics for all the test cases
            end
            save(sets.name,'perf_rst','fullTable','sets'); % Save partial results
            if strcmp(sets.proc_type,'GPU')
                reset(gpuDevice);
            end
            
        end
    end
end

%% Save total results
save('Comparison-rst','fullTable');
fprintf('\n\nA total of %i time experiments was executed!\n',it)
fprintf('Date and time at the beginning of the process: \t%s \n',datestr(t));
fprintf('Date and time at the end of the process: \t%s\n\n',datestr(now));
cd ..
