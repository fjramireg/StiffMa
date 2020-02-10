function WriteEStiffPerfScript(sets)
% Writes a script to measure the performance of the code using "runperf"

nameFile = [sets.name,'.m'];
fileID = fopen(nameFile,'w');
fprintf(fileID,'sets.nel = %d;\n',sets.nel);
fprintf(fileID,"sets.dTE = '%s';\n",sets.dTE);
fprintf(fileID,"sets.dTN = '%s';\n",sets.dTN);
fprintf(fileID,'[Mesh.elements, Mesh.nodes] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);\n');
fprintf(fileID,'sets.nel = %d;\n',sets.nel^3);

if strcmp(sets.prob_type,'Scalar')
    fprintf(fileID,"sets.sz = %d;\n",36);
    fprintf(fileID,"sets.edof = %d;\n",8);
    fprintf(fileID,"c = %d;\n\n",1);
    
    if strcmp(sets.proctype,'CPU')
        fprintf(fileID,'%s\n','%% Element stiffness matrix computation on CPU (Scalar)');
        fprintf(fileID,'Ke = eStiff_ssa(Mesh, c, sets);\n\n');
        fprintf(fileID,'%s\n','%% Element stiffness matrix computation on CPU (Scalar-Symmetry)');
        fprintf(fileID,'Ke = eStiff_sssa(Mesh, c, sets);\n\n');
        
    elseif strcmp(sets.proctype,'GPU')
        fprintf(fileID,"d = gpuDevice;\n");
        fprintf(fileID,"sets.tbs = d.MaxThreadsPerBlock;\n");
        fprintf(fileID,"sets.numSMs   = d.MultiprocessorCount;\n");
        fprintf(fileID,"sets.WarpSize = d.SIMDWidth;\n");
        fprintf(fileID,"elementsGPU = gpuArray(elements');\n");
        fprintf(fileID,"nodedGPU = gpuArray(nodes');\n");
        fprintf(fileID,'%s\n','%% Element stiffness matrix computation on GPU (Scalar-Symmetry)');
        fprintf(fileID,'Ke = eStiff_spsa(elementsGPU, nodesGPU, c, sets);\n');
        fprintf(fileID,'wait(d);\n');
        
    else
        error('Error. No processor type defined.');
    end
    
    
    
elseif strcmp(sets.prob_type,'Vector')
    fprintf(fileID,"sets.sz = %d;\n",300);
    fprintf(fileID,"sets.edof = %d;\n",24);
    fprintf(fileID,"MP.E = %d;\n",200e9);
    fprintf(fileID,"MP.nu = %d;\n\n",0.3);
    
    if strcmp(sets.proctype,'CPU')
        fprintf(fileID,'%s\n','%% Element stiffness matrix computation on CPU (Vector)');
        fprintf(fileID,'Ke = eStiff_vsa(Mesh, MP, sets);\n\n');
        fprintf(fileID,'%s\n','%% Element stiffness matrix computation on CPU (Vector-Symmetry)');
        fprintf(fileID,'Ke = eStiff_vssa(Mesh, MP, sets);\n\n');
        
    elseif strcmp(sets.proctype,'GPU')
        fprintf(fileID,"d = gpuDevice;\n");
        fprintf(fileID,"sets.tbs = d.MaxThreadsPerBlock;\n");
        fprintf(fileID,"sets.numSMs   = d.MultiprocessorCount;\n");
        fprintf(fileID,"sets.WarpSize = d.SIMDWidth;\n");
        fprintf(fileID,"elementsGPU = gpuArray(elements');\n");
        fprintf(fileID,'%s\n','%% Element stiffness matrix computation on GPU (Vector)');
        fprintf(fileID,'[iK, jK] = Index_vps(elementsGPU, sets);\n');
        fprintf(fileID,'wait(d);\n');
        
    else
        error('Error. No processor type defined.');
    end
    
else
    error('Error. No problem type defined.');
end

fclose(fileID);
