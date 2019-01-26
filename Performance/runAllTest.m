% Adding folders to the path
addpath('../Scalar/');
addpath('../Common');

% Platform details
MWver = ver;    % Version information for MathWorks products
Matlab_v = version; % Version number for MATLAB and libraries
platform = system_dependent('getos');
if ismac
    pf = 'MAC_';     % Code to run on Mac platform
elseif isunix
    pf = 'LNX_';    % Code to run on Linux platform
elseif ispc
    pf = 'WIN_';     % Code to run on Windows platform
else
    error('Platform not supported');
end

% Variables for performance tests
dTEall = {'int32','uint32','int64','uint64','double'};
dTNall = {'single','double'};
nel_all= [5,10];
procall = {'CPU','GPU'};

% Runs all tests
for pt = 1:2
    proctype = procall{pt};
    
    for i = 1:length(dTEall)
        dTE = dTEall{i};
        
        for j = 1:length(dTNall)
            dTN = dTNall{j};
            
            for k = 1:length(nel_all)
                nel = nel_all(k);
                
                WrittingPerfFn(dTE,dTN,proctype,nel);
                pause(1);
                fprintf("\n\nStarting the performance measurement with the following parameters:\n");
                fprintf("Date type for 'elements': '%s'\n",dTE);
                fprintf("Date type for 'nodes': '%s'\n",dTN);
                fprintf("Number of finine elements: %dx%dx%d (%d)\n",nel,nel,nel,nel^3);
                fprintf("Processor type: '%s'\n\n",proctype);
                perf_rst = runperf('Scalar_perftest');
                name = [pf,proctype,'_N',dTN,'_E',dTE,'_nel',num2str(nel)];
                save(name,'perf_rst','dTE','dTN','nel','proctype','Matlab_v','platform');
                
            end
        end
    end
end
