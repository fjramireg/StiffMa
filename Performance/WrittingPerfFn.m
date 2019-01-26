function WrittingPerfFn(dTE,dTN,proctype,nel)

fileID = fopen('Scalar_perftest.m','w');

if strcmp(proctype,'CPU')
    
    fprintf(fileID,'nel = %d;\n',nel);
    fprintf(fileID,"dTE = '%s';\n",dTE);
    fprintf(fileID,"dTN = '%s';\n",dTN);
    fprintf(fileID,'[elements, nodes] = CreateMesh(nel,nel,nel,dTE,dTN,0,0);\n');
    fprintf(fileID,'c = 1;\n');
    fprintf(fileID,'N = size(nodes,1);\n');
    fprintf(fileID,'[iK, jK] = IndexScalarSymCPU(elements);\n');
    fprintf(fileID,'Ke = Hex8scalarSymCPU(elements,nodes,c);\n\n');
    fprintf(fileID,'%s\n','%% Index computation on CPU');
    fprintf(fileID,'[i, j] = IndexScalarSymCPU(elements);\n\n');
    fprintf(fileID,'%s\n','%% Element stiffness matrices computation on CPU');
    fprintf(fileID,'v = Hex8scalarSymCPU(elements,nodes,c);\n\n');
    if strcmp(dTN,'double')
        fprintf(fileID,'%s\n','%% Assembly of global sparse matrix on CPU');
        fprintf(fileID,'K = AssemblyStiffMat(iK,jK,Ke,N,dTE,dTN);\n');
    end
    
elseif strcmp(proctype,'GPU')
    
    fprintf(fileID,'nel = %d;\n',nel);
    fprintf(fileID,"dTE = '%s';\n",dTE);
    fprintf(fileID,"dTN = '%s';\n",dTN);
    fprintf(fileID,'[elements, nodes] = CreateMesh(nel,nel,nel,dTE,dTN,0,0);\n');
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

fclose(fileID);
