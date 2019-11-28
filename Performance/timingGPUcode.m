%  Performance measurement of the SCALAR/VECTOR problem using gputimeit

%% Path to run code
addpath('../Common');
addpath('../Scalar/');
addpath('../Vector/');

%% Platform details
sys_info = evalc('configinfo'); % Gets system information
gpudev = gpuDevice;             % Gets GPU basic information

%% Variables for performance tests
nel_all  = [50];            % Cases for mesh size
dTEall   = {'int32'};% Cases for "element" data type
dTNall   = {'double'};                         % Cases for "nodes" data type
proc_all = {'GPU'};                                     % Cases for processor type
prob_all = {'Scalar','Vector'};                         % Cases for problem type

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
                        [elements, nodes] = CreateMesh(nel,nel,nel,dTE,dTN,0,0);% Mesh
                        c = 1;                                                  % Mat properties
                        
                        % Transfer to GPU memory (timing)
                        fh_mems = @() Host2DeviceMemTranf(elements,nodes);      % Handle function
                        rt_mems = gputimeit(fh_mems,2);                         % timing with 2 outputs
                        
                        % Transfer to GPU memory (no timing)
                        elementsGPU=gpuArray(elements'); nodesGPU=gpuArray(nodes');
                        
                        % Index computation on GPU (timing)
                        fh_inds = @() IndexScalarSymGPU(elementsGPU);
                        rt_inds = gputimeit(fh_inds,2);
                        
                        % Element stiffness matrices computation on GPU (timing)
                        fh_Ke_s = @() Hex8scalarSymGPU(elementsGPU,nodesGPU,c);
                        rt_Ke_s = gputimeit(fh_Ke_s,1);
                        
                        % Assembly of global sparse matrix on GPU (timing)
                        if (strcmp(dTN,'double'))
                            N = size(nodes,1);
                            [iK, jK] = IndexScalarSymGPU(elementsGPU);
                            Ke = Hex8scalarSymGPU(elementsGPU,nodesGPU,c);
                            clear elementsGPU nodesGPU;
                            fh_Kg_s = @()  AssemblyStiffMat(iK,jK,Ke,N,dTE,dTN);
                            rt_Kg_s = gputimeit(fh_Kg_s,1);
                        end
                        
                        %TODO: Stores the results
                        
                        % Reset GPU memory (no timing)
                        reset(gpuDevice);
                        
                    elseif strcmp(prob_type,'Vector')
                        
                        % Screen messeges
                        fprintf("\n\nStarting the performance measurement with the following parameters:\n");
                        fprintf("Number of finine elements: %dx%dx%d (%d)\n",nel,nel,nel,nel^3);
                        fprintf("Date type for 'elements': '%s'\n",dTE);
                        fprintf("Date type for 'nodes': '%s'\n",dTN);
                        fprintf("Processor type: '%s'\n",proctype);
                        fprintf("Problem type: '%s'\n\n",prob_type);
                        
                        % Initial setup (no timing)
                        [elements, nodes] = CreateMesh(nel,nel,nel,dTE,dTN,0,0);% Mesh
                        E = 200e9; nu = 0.3;                                    % Mat properties
                        
                        % Transfer to GPU memory (timing)
                        fh_memv = @() Host2DeviceMemTranf(elements,nodes);      % Handle function
                        rt_memv = gputimeit(fh_memv,2);                         % timing with 2 outputs
                        
                        % Transfer to GPU memory (no timing)
                        elementsGPU=gpuArray(elements'); nodesGPU=gpuArray(nodes');
                        
                        % Index computation on GPU (timing)
                        fh_indv = @() IndexVectorSymGPU(elementsGPU);
                        rt_indv = gputimeit(fh_indv,2);
                        
                        % Element stiffness matrices computation on GPU (timing)
                        fh_Ke_v = @() Hex8vectorSymGPU(elementsGPU,nodesGPU,E,nu);
                        rt_Ke_v = gputimeit(fh_Ke_v,1);
                        
                        % Assembly of global sparse matrix on GPU (timing)
                        if (strcmp(dTN,'double'))
                            N = size(nodes,1);
                            [iK, jK] = IndexVectorSymGPU(elementsGPU);
                            Ke = Hex8vectorSymGPU(elementsGPU,nodesGPU,E,nu);
                            clear elementsGPU nodesGPU; %TODO: It seems that the GPU memory is not free after these commands
                            %TODO: Deal with the error "radix_sort: failed to get memory buffer"
                            try
                                AssemblyStiffMat(iK,jK,Ke,3*N,dTE,dTN);
                            catch ME
                                %                                 if (strcmp(ME.identifier,'MATLAB:catenate:dimensionMismatch'))
                                rethrow(ME)
                            end
%                             fh_Kg_v = @()  AssemblyStiffMat(iK,jK,Ke,3*N,dTE,dTN);
%                             rt_Kg_v = gputimeit(fh_Kg_v,1);
                        end
                        
                         %TODO: Stores the results
                        
                        % Reset GPU memory (no timing)
                        reset(gpuDevice);
                        
                    else
                        error('No problem type defined!');
                        
                    end
                    
                end
            end
        end
    end
end
