function WriteAssemblyPerfScript(sets)
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
fprintf(fileID,'[Mesh.elements, Mesh.nodes] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets.dTE,sets.dTN);\n');
fprintf(fileID,'sets.nel = %d;\n',sets.nel^3);
fprintf(fileID,"d = gpuDevice;\n");
fprintf(fileID,"sets.tbs = d.MaxThreadsPerBlock;\n");
fprintf(fileID,"sets.numSMs   = d.MultiprocessorCount;\n");
fprintf(fileID,"sets.WarpSize = d.SIMDWidth;\n");
fprintf(fileID,"elementsGPU = gpuArray(Mesh.elements');\n");
fprintf(fileID,"nodesGPU = gpuArray(Mesh.nodes');\n");

% 'Scalar'
if strcmp(sets.prob_type,'Scalar')
    fprintf(fileID,"sets.sz = %d;\n",36);
    fprintf(fileID,"sets.edof = %d;\n",8);
    fprintf(fileID,"c = %d;\n",1);
    fprintf(fileID,'[iKd, jKd] = Index_spsa(elementsGPU, sets);\n');
    fprintf(fileID,'Ked = eStiff_spsa(elementsGPU, nodesGPU, c, sets);\n');
    fprintf(fileID,'clear elementsGPU nodesGPU;\n');
    fprintf(fileID,'iKs = gather(iKd);\n');
    fprintf(fileID,'jKs = gather(jKd);\n');
    fprintf(fileID,'Kes = gather(Ked);\n');
    fprintf(fileID,"[iK, jK] = Index_ssa(Mesh.elements, sets);\n");
    fprintf(fileID,'Ke = eStiff_ssa(Mesh, c, sets);\n');
    
    % 'Scalar'-'CPU'
    fprintf(fileID,'\n%s\n','%% Assembly-CPU-Scalar');
    fprintf(fileID,'K = AssemblyStiffMa(iK, jK, Ke, sets.dTE, sets.dTN);\n');
    
    % 'Scalar'-'CPU'-'Symmetry'
    fprintf(fileID,'\n%s\n','%% Assembly-CPU-Scalar-Symmetry');
    fprintf(fileID,'K = AssemblyStiffMa(iKs, jKs, Kes, sets.dTE, sets.dTN);\n');
    
    % 'Scalar'-'GPU'-'Symmetry'
    fprintf(fileID,'\n%s\n','%% Assembly-GPU-Scalar-Symmetry');
    fprintf(fileID,'K = AssemblyStiffMa(iKd, jKd, Ked, sets.dTE, sets.dTN);\n');
    fprintf(fileID,'wait(d);\n');
    
    
    % 'Vector'
elseif strcmp(sets.prob_type,'Vector')
    fprintf(fileID,"sets.sz = %d;\n",300);
    fprintf(fileID,"sets.edof = %d;\n",24);
    fprintf(fileID,"MP.E = %d;\n",200e9);
    fprintf(fileID,"MP.nu = %d;\n",0.3);
    fprintf(fileID,'[iKd, jKd] = Index_vpsa(elementsGPU, sets);\n');
    fprintf(fileID,'Ked = eStiff_vpsa(elementsGPU, nodesGPU, MP, sets);\n');
    fprintf(fileID,'clear elementsGPU nodesGPU;\n');
    fprintf(fileID,'iKs = gather(iKd);\n');
    fprintf(fileID,'jKs = gather(jKd);\n');
    fprintf(fileID,'Kes = gather(Ked);\n');
    fprintf(fileID,"[iK, jK] = Index_vsa(Mesh.elements, sets);\n");
    fprintf(fileID,'Ke = eStiff_vsa(Mesh, MP, sets);\n');
    
    % 'Vector'-'CPU'
    fprintf(fileID,'\n%s\n','%% Assembly-CPU-Vector');
    fprintf(fileID,'K = AssemblyStiffMa(iK, jK, Ke, sets.dTE, sets.dTN);\n');
    
    % 'Vector'-'CPU'-'Symmetry'
    fprintf(fileID,'\n%s\n','%% Assembly-CPU-Vector-Symmetry');
    fprintf(fileID,'K = AssemblyStiffMa(iKs, jKs, Kes, sets.dTE, sets.dTN);\n');
    
    % 'Vector'-'GPU'-'Symmetry'
    fprintf(fileID,'\n%s\n','%% Assembly-GPU-Vector-Symmetry');
    fprintf(fileID,'K = AssemblyStiffMa(iKd, jKd, Ked, sets.dTE, sets.dTN);\n');
    fprintf(fileID,'wait(d);\n');
    
else
    error('Error. No problem type defined.');
end

fclose(fileID);
