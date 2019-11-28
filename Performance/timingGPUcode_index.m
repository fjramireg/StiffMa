%  Performance measurement of the SCALAR/VECTOR problem using gputimeit

%% Path to run code
addpath('../Common');
addpath('../Scalar/');
addpath('../Vector/');

%% Platform details
sys_info = evalc('configinfo'); % Write system information
d = gpuDevice();

%% Variables for performance tests
nel_all  = [90];            % Cases for mesh size
dTEall   = {'int32'};% Cases for "element" data type
dTNall   = {'double'};                         % Cases for "nodes" data type
proc_all = {'GPU'};                                     % Cases for processor type
prob_all = {'Scalar','Vector'};                         % Cases for problem type
ncases   = length(nel_all)*length(dTEall)*length(dTNall)*length(proc_all)*length(prob_all);

%% Runs all possible tests
for k = 1:length(nel_all)
    nel = nel_all(k);
    
    for i = 1:length(dTEall)
        dTE = dTEall{i};
        
        for j = 1:length(dTNall)
            dTN = dTNall{j};
            if (strcmp(dTN,'single') && (strcmp(dTE,'int64')...
                    || strcmp(dTE,'uint64') || strcmp(dTE,'double')))   % ommit this cases
                continue;
            end
            
            for pt = 1:length(proc_all)
                proctype = proc_all{pt};
                
                for pbl = 1:length(prob_all)
                    prob_type = prob_all{pbl};
                    
                    if strcmp(prob_type,'Scalar')
                        
                        % Screen messeges
                        fprintf("\n\nStarting the performance measurement with the following parameters:\n");
                        fprintf("Number of finine elements: %dx%dx%d (%d)\n",nel,nel,nel,nel^3);
                        fprintf("Date type for 'elements': '%s'\n",dTE);
                        fprintf("Date type for 'nodes': '%s'\n",dTN);
                        fprintf("Processor type: '%s'\n",proctype);
                        fprintf("Problem type: '%s'\n\n",prob_type);
                        
                        % Initial setup (no timing)
                        [elements, ~] = CreateMesh(nel,nel,nel,dTE,dTN,0,0);% Mesh
                        fprintf('Used GPU memory (start): %0.2f GB\n',(d.TotalMemory - d.AvailableMemory)/1e9);
                        
                        % Transfer to GPU memory (no timing)
                        elementsGPU=gpuArray(elements');
                        wait(d);
                        fprintf('Used GPU memory (memtr): %0.2f GB\n',(d.TotalMemory - d.AvailableMemory)/1e9);
                        
                        % Index computation on GPU (timing)
                        fh_inds = @() IndexScalarSymGPU(elementsGPU);
                        rt_inds = gputimeit(fh_inds,2);
                        wait(d);
                        fprintf('Used GPU memory (index): %0.2f GB\n',(d.TotalMemory - d.AvailableMemory)/1e9);
                        
                        %TODO: Stores the results
                        
                    elseif strcmp(prob_type,'Vector')
                        
                        % Screen messeges
                        fprintf("\n\nStarting the performance measurement with the following parameters:\n");
                        fprintf("Number of finine elements: %dx%dx%d (%d)\n",nel,nel,nel,nel^3);
                        fprintf("Date type for 'elements': '%s'\n",dTE);
                        fprintf("Date type for 'nodes': '%s'\n",dTN);
                        fprintf("Processor type: '%s'\n",proctype);
                        fprintf("Problem type: '%s'\n\n",prob_type);
                        
                        % Initial setup (no timing)
                        [elements, ~] = CreateMesh(nel,nel,nel,dTE,dTN,0,0);% Mesh
                        fprintf('Used GPU memory (start): %0.2f GB\n',(d.TotalMemory - d.AvailableMemory)/1e9);
                        
                        % Transfer to GPU memory (no timing)
                        elementsGPU=gpuArray(elements');
                        wait(d);
                        fprintf('Used GPU memory (memtr): %0.2f GB\n',(d.TotalMemory - d.AvailableMemory)/1e9);
                        
                        % Index computation on GPU (timing)
                        try %TODO: How to catch the error "Out of memory on device"
                            [~,~]=IndexVectorSymGPU(elementsGPU);
                        catch ME
                            rethrow(ME)
                        end
%                         fh_indv = @() IndexVectorSymGPU(elementsGPU);
%                         rt_indv = gputimeit(fh_indv,2);
%                         wait(d);
%                         fprintf('Used GPU memory (index): %0.2f GB\n',(d.TotalMemory - d.AvailableMemory)/1e9);
%                         
                        %TODO: Stores the results
                        
                    else
                        error('No problem type defined!');
                        
                    end
                    
                    % Reset GPU memory (no timing)
                    reset(gpuDevice); %pause(10);
                    fprintf('Used GPU memory (_end_): %0.2f GB\n',(d.TotalMemory - d.AvailableMemory)/1e9);
                    
                end
            end
        end
    end
end
