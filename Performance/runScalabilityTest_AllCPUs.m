% Runs all code parts to assembly the stiffness matrix on CPU serial
% version testing all possibilities and measuring their performance by
% using the built-in function 'runperf'

%% Adding folders to the path
addpath('../Scalar/');
addpath('../Vector/');
addpath('../Common');

%% Platform details
sys_info = evalc('configinfo'); % System information
info_CPU = cpuinfo;             % CPU information
info_GPU = gpuinfo;             % GPU information
if ismac
    pf = 'MAC_';    % Code to run on Mac platform
elseif isunix
    pf = 'LNX_';    % Code to run on Linux platform
elseif ispc
    pf = 'WIN_';    % Code to run on Windows platform
else
    error('Platform not supported');
end

%% Variables for performance tests
nel_all  = [10,20,30,40,50,60,70,80,90,100];            % Cases for mesh size. nel: number of finite element in each dir.
dTEall   = {'int32','uint32','int64','uint64','double'};% Cases for "element" data type
dTNall   = {'single','double'};                         % Cases for "nodes" data type
code_all = {'CPUs'};                                    % Cases for code type. CPUs: CPU serial code
prob_all = {'Scalar','Vector'};                         % Cases for problem type

%% Runs all tests
for i = 1:length(nel_all)
    nel = nel_all(i);
    
    for j = 1:length(dTEall)
        dTE = dTEall{j};
        
        for k = 1:length(dTNall)
            dTN = dTNall{k};
            if (strcmp(dTN,'single') && (strcmp(dTE,'int64')...
                    || strcmp(dTE,'uint64') || strcmp(dTE,'double')))
                continue;                               % ommit this cases
            end
            
            for pbl = 1:length(prob_all)
                prob_type = prob_all{pbl};
                
                for ct = 1:length(code_all)
                    codetype = code_all{ct};
                    
                    WrittingPerfFn(nel,dTE,dTN,codetype,prob_type,pf);
                    fprintf("\n\nStarting the performance measurement with the following parameters:\n");
                    fprintf("Number of finine elements: %dx%dx%d (%d)\n",nel,nel,nel,nel^3);
                    fprintf("Date type for 'elements': '%s'\n",dTE);
                    fprintf("Date type for 'nodes': '%s'\n",dTN);
                    fprintf("Problem type: '%s'\n",prob_type);
                    fprintf("Type of code: '%s'\n",codetype);
                    fprintf("CPU processor: '%s'\n",info_CPU.Name);
                    fprintf("GPU co-processor: '%s'\n\n",info_GPU.Name);
                    nameFile = ['perftest',pf,prob_type(1),'_',codetype,'.m'];
                    perf_rst = runperf(nameFile);
                    nameFolder = [pf,codetype,'Rst_perf'];
                    if mkdir(nameFolder)
                        nameData = [nameFolder,'/',pf,prob_type(1),'_',codetype,'_N',dTN,'_E',dTE,'_nel',num2str(nel)];
                        save nameData perf_rst dTE dTN nel codetype prob_type sys_info info_CPU info_GPU;
                    end
                    
                end
            end
        end
    end
end
