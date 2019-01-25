%  Performance measure of the SCALAR problem

% General setup
addpath('../Scalar/');
addpath('../Common');

dTEall = {'int32','uint32','int64','uint64','double'};  % Data precision for "elements"
dTNall = {'single','double'};                           % Data precision for "nodes"
nel_all= [1,2,3];

for i = 1:length(dTEall)
    dTE = dTEall{i};
    
    for j = 1:length(dTNall)
        dTN = dTNall{j};
        
        for k = 1:length(nel_all)                                    % Number of elements on each direction
            
            % Mesh creation (no timing)
            nel = nel_all(k);
            [elements, nodes] = CreateMesh(nel,nel,nel,dTE,dTN,0,0); % Mesh creation
                        
            % Index creation (timing)
            fh_ind_h = @() IndexScalarSymCPU(elements); % Handle to function Index on CPU
            rt_ind_h = timeit(fh_ind_h,2);              % timing with 2 outputs
            
            % Element stiffness matrices computation (timing)
            c = 1;                                      % Material properties
            fh_Ke_h = @() Hex8scalarSymCPU(elements,nodes,c);
            rt_Ke_h = timeit(fh_Ke_h,1);
            
            % Assembly of global stiffness matrix timing
            if (strcmp(dTN,'double'))
                N = size(nodes,1);                      % Total number of nodes (DOFs)
                [iK, jK] = IndexScalarSymCPU(elements);
                Ke = Hex8scalarSymCPU(elements,nodes,c);
                fh_Kg_h = @()  AssemblyStiffMat(iK,jK,Ke(:),N,dTE,dTN);
                rt_Kg_h = timeit(fh_Kg_h,1);
            end
            
        end
    end
end
