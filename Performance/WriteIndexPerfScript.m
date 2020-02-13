function WriteIndexPerfScript(sets)
% Writes a script to measure the performance of the code using "runperf"

%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.
%
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  12/02/2020. Version: 1.4

Filename = [sets.name,'.m'];
fileID = fopen(Filename,'w');
fprintf(fileID,'sets.nel = %d;\n',sets.nel);
fprintf(fileID,"sets.dTE = '%s';\n",sets.dTE);
fprintf(fileID,"sets.dTN = '%s';\n",sets.dTN);
fprintf(fileID,'[elements, ~] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);\n');
fprintf(fileID,'sets.nel = %d;\n',sets.nel^3);

%% 'Scalar'
if strcmp(sets.prob_type,'Scalar')
    fprintf(fileID,"sets.sz = %d;\n",36);
    fprintf(fileID,"sets.edof = %d;\n",8);    
    
    % 'Scalar'-'CPU'
    if strcmp(sets.proc_type,'CPU')
        fprintf(fileID,'\n%s\n','%% Index-CPU-Scalar');
        fprintf(fileID,'[iK, jK] = Index_ssa(elements, sets);\n');
        
        % 'Scalar'-'CPU'-'Symmetry'
        fprintf(fileID,'\n%s\n','%% Index-CPU-Scalar-Symmetry');
        fprintf(fileID,'[iK, jK] = Index_sssa(elements, sets);\n');
        
        % 'Scalar'-'GPU'-'Symmetry'
    elseif strcmp(sets.proc_type,'GPU')
        fprintf(fileID,"d = gpuDevice;\n");
        fprintf(fileID,"sets.tbs = d.MaxThreadsPerBlock;\n");
        fprintf(fileID,"sets.numSMs   = d.MultiprocessorCount;\n");
        fprintf(fileID,"sets.WarpSize = d.SIMDWidth;\n");
        fprintf(fileID,"elementsGPU = gpuArray(elements');\n");
        fprintf(fileID,'\n%s\n','%% Index-GPU-Scalar-Symmetry');
        fprintf(fileID,'[iKd, jKd] = Index_spsa(elementsGPU, sets);\n');
        fprintf(fileID,'wait(d);\n');
        
    else
        error('Error. No processor type defined.');
    end
    
    
    %% 'Vector'
elseif strcmp(sets.prob_type,'Vector')
    fprintf(fileID,"sets.edof = %d;\n",24);
    fprintf(fileID,"sets.sz = %d;\n",300);
    
    % 'Vector'-'CPU'
    if strcmp(sets.proc_type,'CPU')
        fprintf(fileID,'\n%s\n','%% Index-CPU-Vector');
        fprintf(fileID,'[iK, jK] = Index_vsa(elements, sets);\n');
        
        % 'Vector'-'CPU'-'Symmetry'
        fprintf(fileID,'\n%s\n','%% Index-CPU-Vector-Symmetry');
        fprintf(fileID,'[iK, jK] = Index_vssa(elements, sets);\n');
        
        % 'Vector'-'GPU'-'Symmetry'
    elseif strcmp(sets.proc_type,'GPU')
        fprintf(fileID,"d = gpuDevice;\n");
        fprintf(fileID,"sets.tbs = d.MaxThreadsPerBlock;\n");
        fprintf(fileID,"sets.numSMs   = d.MultiprocessorCount;\n");
        fprintf(fileID,"sets.WarpSize = d.SIMDWidth;\n");
        fprintf(fileID,"elementsGPU = gpuArray(elements');\n");
        fprintf(fileID,'\n%s\n','%% Index-GPU-Vector-Symmetry');
        fprintf(fileID,'[iKd, jKd] = Index_vpsa(elementsGPU, sets);\n');
        fprintf(fileID,'wait(d);\n');
        
    else
        error('Error. No processor type defined.');
    end
    
else
    error('Error. No problem type defined.');
end

fclose(fileID);
