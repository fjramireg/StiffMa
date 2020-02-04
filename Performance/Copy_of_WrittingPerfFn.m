function WrittingPerfFn(nel,dTE,dTN,proctype,prob_type,platform)
% Writes a script to measure the performance of the code using "runperf"

nameFile = ['perftest',platform,prob_type(1),'_',proctype,'.m'];
fileID = fopen(nameFile,'w');

if strcmp(prob_type,'Scalar')
    
    if strcmp(proctype,'CPU')        
        fprintf(fileID,'nel = %d;\n',nel);
        fprintf(fileID,"dTE = '%s';\n",dTE);
        fprintf(fileID,"dTN = '%s';\n",dTN);
        fprintf(fileID,'[elements, nodes] = CreateMesh(nel,nel,nel,dTE,dTN);\n');
        fprintf(fileID,'c = 1;\n');
        fprintf(fileID,'N = size(nodes,1);\n');
        fprintf(fileID,'[iK, jK] = Index_sss(elements, sets);\n');
        fprintf(fileID,'Ke = Hex8scalarSymCPU(elements,nodes,c);\n\n');
        fprintf(fileID,'%s\n','%% Index computation on CPU (serial)');
        fprintf(fileID,'[i, j] = IndexScalarSymCPU(elements);\n\n');
        fprintf(fileID,'%s\n','%% Element stiffness matrices computation on CPU (serial)');
        fprintf(fileID,'v = Hex8scalarSymCPU(elements,nodes,c);\n\n');
        if strcmp(dTN,'double')
            fprintf(fileID,'%s\n','%% Assembly of global sparse matrix on CPU');
            fprintf(fileID,'K = AssemblyStiffMat(iK,jK,Ke,N,dTE,dTN);\n');
        end        
        
    elseif strcmp(proctype,'GPU')        
        fprintf(fileID,'nel = %d;\n',nel);
        fprintf(fileID,"dTE = '%s';\n",dTE);
        fprintf(fileID,"dTN = '%s';\n",dTN);
        fprintf(fileID,'[elements, nodes] = CreateMesh(nel,nel,nel,dTE,dTN);\n');
        fprintf(fileID,'c = 1;\n');
        fprintf(fileID,'N = size(nodes,1);\n');
        fprintf(fileID,"elementsGPU = gpuArray(elements');\n");
        fprintf(fileID,"nodesGPU    = gpuArray(nodes');\n");
        fprintf(fileID,"[iK, jK] = IndexScalarSymGPU(elementsGPU);\n");
        fprintf(fileID,"Ke = Hex8scalarSymGPU(elementsGPU,nodesGPU,c);\n\n");
        fprintf(fileID,'%s\n','%% Transfer to GPU memory');
        fprintf(fileID,"eGPU = gpuArray(elements');\n");
        fprintf(fileID,"nGPU = gpuArray(nodes');\n\n");
        fprintf(fileID,'%s\n','%% Index computation on GPU');
        fprintf(fileID,'[i, j] = IndexScalarSymGPU(elementsGPU);\n\n');
        fprintf(fileID,'%s\n','%% Element stiffness matrices computation on GPU');
        fprintf(fileID,'v = Hex8scalarSymGPU(elementsGPU,nodesGPU,c);\n\n');
        if strcmp(dTN,'double')
            fprintf(fileID,'%s\n','%% Assembly of global sparse matrix on GPU');
            fprintf(fileID,'K = AssemblyStiffMat(iK,jK,Ke,N,dTE,dTN);\n');
        end
        
    else
        error('Error. No processor type defined.');
    end
  
    
    
elseif strcmp(prob_type,'Vector')
    
    if strcmp(proctype,'CPU')
        
        fprintf(fileID,'nel = %d;\n',nel);
        fprintf(fileID,"dTE = '%s';\n",dTE);
        fprintf(fileID,"dTN = '%s';\n",dTN);
        fprintf(fileID,'[elements, nodes] = CreateMesh(nel,nel,nel,dTE,dTN);\n');
        fprintf(fileID,'E = 200e9;\n');
        fprintf(fileID,'nu = 0.3;\n');
        fprintf(fileID,'N = size(nodes,1);\n');
        fprintf(fileID,'[iK, jK] = IndexVectorSymCPU(elements);\n');
        fprintf(fileID,'Ke = Hex8vectorSymCPU(elements,nodes,E,nu);\n\n');
        fprintf(fileID,'%s\n','%% Index computation on CPU (serial)');
        fprintf(fileID,'[i, j] = IndexVectorSymCPU(elements);\n\n');
        fprintf(fileID,'%s\n','%% Element stiffness matrices computation on CPU (serial)');
        fprintf(fileID,'v = Hex8vectorSymCPU(elements,nodes,E,nu);\n\n');
        if strcmp(dTN,'double')
            fprintf(fileID,'%s\n','%% Assembly of global sparse matrix on CPU');
            fprintf(fileID,'K = AssemblyStiffMat(iK,jK,Ke,3*N,dTE,dTN);\n');
        end
        
        
    elseif strcmp(proctype,'GPU')
        
        fprintf(fileID,'nel = %d;\n',nel);
        fprintf(fileID,"dTE = '%s';\n",dTE);
        fprintf(fileID,"dTN = '%s';\n",dTN);
        fprintf(fileID,'[elements, nodes] = CreateMesh(nel,nel,nel,dTE,dTN);\n');
        fprintf(fileID,'E = 200e9;\n');
        fprintf(fileID,'nu = 0.3;\n');
        fprintf(fileID,'N = size(nodes,1);\n');
        fprintf(fileID,"elementsGPU = gpuArray(elements');\n");
        fprintf(fileID,"nodesGPU    = gpuArray(nodes');\n");
        fprintf(fileID,"[iK, jK] = IndexVectorSymGPU(elementsGPU);\n");
        fprintf(fileID,"Ke = Hex8vectorSymGPU(elementsGPU,nodesGPU,E,nu);\n\n");
        fprintf(fileID,'%s\n','%% Transfer to GPU memory');
        fprintf(fileID,"eGPU = gpuArray(elements');\n");
        fprintf(fileID,"nGPU = gpuArray(nodes');\n\n");
        fprintf(fileID,'%s\n','%% Index computation on GPU');
        fprintf(fileID,'[i, j] = IndexVectorSymGPU(elementsGPU);\n\n');
        fprintf(fileID,'%s\n','%% Element stiffness matrices computation on GPU');
        fprintf(fileID,'v = Hex8vectorSymGPU(elementsGPU,nodesGPU,E,nu);\n\n');
        if strcmp(dTN,'double')
            fprintf(fileID,'%s\n','%% Assembly of global sparse matrix on GPU');
            fprintf(fileID,'K = AssemblyStiffMat(iK,jK,Ke,3*N,dTE,dTN);\n');
        end
        
    else
        error('Error. No processor type defined.');
    end
    
else
    error('Error. No problem type defined.');
end

fclose(fileID);
