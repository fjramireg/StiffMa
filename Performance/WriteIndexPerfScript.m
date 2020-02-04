function WriteIndexPerfScript(sets)
% Writes a script to measure the performance of the code using "runperf"

nameFile = [sets.name,'.m'];
fileID = fopen(nameFile,'w');
fprintf(fileID,'sets.nel = %d;\n',sets.nel);
fprintf(fileID,"sets.dTE = '%s';\n",sets.dTE);
fprintf(fileID,"sets.dTN = '%s';\n",sets.dTN);
fprintf(fileID,'[elements, ~] = CreateMesh(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);\n');


if strcmp(sets.prob_type,'Scalar')
    fprintf(fileID,"sets.sz = %d;\n\n",36);
    
    if strcmp(sets.proctype,'CPU')
        fprintf(fileID,'%s\n','%% Index computation on CPU (Scalar)');
        fprintf(fileID,'[iK, jK] = Index_sss(elements, sets);\n');
        
    elseif strcmp(sets.proctype,'GPU')
        fprintf(fileID,"d = gpuDevice;\n");
        fprintf(fileID,"sets.tbs = d.MaxThreadsPerBlock;\n");
        fprintf(fileID,"sets.numSMs   = d.MultiprocessorCount;\n");
        fprintf(fileID,"sets.WarpSize = d.SIMDWidth;\n");
        fprintf(fileID,"elementsGPU = gpuArray(elements');\n");
        fprintf(fileID,'%s\n','%% Index computation on GPU (Scalar)');
        fprintf(fileID,'[iK, jK] = Index_sps(elementsGPU, sets);\n');
        fprintf(fileID,'wait(gpuDevice);\n');
        
    else
        error('Error. No processor type defined.');
    end
    
    
    
elseif strcmp(sets.prob_type,'Vector')
    fprintf(fileID,"sets.sz = %d;\n",300);
    fprintf(fileID,"sets.edof = %d;\n\n",24);
    
    if strcmp(sets.proctype,'CPU')
        fprintf(fileID,'%s\n','%% Index computation on CPU (Vector)');
        fprintf(fileID,'[iK, jK] = Index_vss(elements, sets);\n');
        
    elseif strcmp(sets.proctype,'GPU')
        fprintf(fileID,"d = gpuDevice;\n");
        fprintf(fileID,"sets.tbs = d.MaxThreadsPerBlock;\n");
        fprintf(fileID,"sets.numSMs   = d.MultiprocessorCount;\n");
        fprintf(fileID,"sets.WarpSize = d.SIMDWidth;\n");
        fprintf(fileID,"elementsGPU = gpuArray(elements');\n");
        fprintf(fileID,'%s\n','%% Index computation on GPU (Vector)');
        fprintf(fileID,'[iK, jK] = Index_vps(elementsGPU, sets);\n');
        fprintf(fileID,'wait(gpuDevice);\n');
        
    else
        error('Error. No processor type defined.');
    end
    
else
    error('Error. No problem type defined.');
end

fclose(fileID);
