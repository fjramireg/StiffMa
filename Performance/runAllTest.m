% Adding folders to the path
addpath('../Scalar/');
addpath('../Vector/');
addpath('../Common');

% Platform details
MWver = ver;        % Version information for MathWorks products
Matlab_v = version; % Version number for MATLAB and libraries
platform = system_dependent('getos');
if ismac
    pf = 'MAC_';    % Code to run on Mac platform
elseif isunix
    pf = 'LNX_';    % Code to run on Linux platform
elseif ispc
    pf = 'WIN_';    % Code to run on Windows platform
else
    error('Platform not supported');
end

% Variables for performance tests
nel_all  = [10,20,40,80,160,320,640];                   % Cases for mesh size
dTEall = {'int32','uint32','int64','uint64','double'};  % Cases for "element" data type
dTNall = {'single','double'};                           % Cases for "nodes" data type
proc_all = {'CPUs','CPUp','GPU'};                       % Cases for processor type
prob_all = {'Scalar','Vector'};                         % Cases for problem type

% Runs all tests
for k = 1:length(nel_all)
    nel = nel_all(k);    
    
    for i = 1:length(dTEall)
        dTE = dTEall{i};
        
        for j = 1:length(dTNall)
            dTN = dTNall{j};
            
            for pt = 1:length(proc_all)
                proctype = proc_all{pt};
                
                for pbl = 1:length(prob_all)
                    prob_type = prob_all{pbl};
                    
                    WrittingPerfFn(nel,dTE,dTN,proctype,prob_type);
                    fprintf("\n\nStarting the performance measurement with the following parameters:\n");
                    fprintf("Number of finine elements: %dx%dx%d (%d)\n",nel,nel,nel,nel^3);
                    fprintf("Date type for 'elements': '%s'\n",dTE);
                    fprintf("Date type for 'nodes': '%s'\n",dTN);
                    fprintf("Processor type: '%s'\n",proctype);
                    fprintf("Problem type: '%s'\n\n",prob_type);
                    perf_rst = runperf('perftest');
                    name = [pf,prob_type(1:3),proctype,'_N',dTN,'_E',dTE,'_nel',num2str(nel)];
                    save(name,'perf_rst','dTE','dTN','nel','proctype','prob_type','Matlab_v','platform');
                    
                end
            end
        end
    end
end
