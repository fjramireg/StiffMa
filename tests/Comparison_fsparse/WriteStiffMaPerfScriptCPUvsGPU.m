function WriteStiffMaPerfScriptCPUvsGPU(sets)
% Writes a script to measure the performance of the code using "runperf"

%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.
%
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Created:  18/02/2020. Version: 1.4

Filename = [sets.name,'.m'];
fileID = fopen(Filename,'w');

% Inputs
fprintf(fileID,'%s\n','% Inputs');
fprintf(fileID,'nel = %d;\n',sets.nel);
fprintf(fileID,'sets.sf = %d;\n',sets.sf);
fprintf(fileID,"sets.dTE = '%s';\n",sets.dTE);
fprintf(fileID,"sets.dTN = '%s';\n",sets.dTN);
fprintf(fileID,"MP.E = 200e9;\n");
fprintf(fileID,"MP.nu = 0.3;\n");
fprintf(fileID,"ct = 384.1;\n");


if strcmp(sets.proc_type, 'GPU')
    
    % Mesh generation
    fprintf(fileID,'\n%s\n','% Mesh generation');
    fprintf(fileID,'[elements, nodes] = CreateMesh2(nel, nel, nel, sets);\n');
    fprintf(fileID,'[sets.nel, sets.nxe]  = size(elements);\n');
    fprintf(fileID,'[sets.nnod, sets.dim] = size(nodes);\n');
    
    % GPU setup
    fprintf(fileID,'\n%s\n', '% GPU setup');
    fprintf(fileID,'dev = gpuDevice;\n');
    fprintf(fileID,'sets.tbs = dev.MaxThreadsPerBlock;\n');
    fprintf(fileID,'sets.numSMs   = dev.MultiprocessorCount;\n');
    fprintf(fileID,'sets.WarpSize = dev.SIMDWidth;\n');
    
    % Memory transfer
    fprintf(fileID,'\n%s\n', '% GPU Memory transfer');
    fprintf(fileID,"elementsGPU = gpuArray(elements');\n");
    fprintf(fileID,"nodesGPU = gpuArray(nodes');\n");
    
    
    if strcmp(sets.prob_type, 'Scalar')
        fprintf(fileID,'sets.dxn = 1;\n');
        sets.dxn = 1;
        fprintf(fileID,'sets.edof = sets.dxn * sets.nxe;\n');
        fprintf(fileID,'sets.sz = (sets.edof * (sets.edof + 1) )/2;\n');
        fprintf(fileID,'sets.tdofs = sets.nnod * sets.dxn;\n');
        
        % For assembly
        fprintf(fileID,'\n%s\n', '% For GPU assembly');
        fprintf(fileID,'[iKd, jKd] = Index_spsa(elementsGPU, sets);\n');
        fprintf(fileID,'Ked = eStiff_spsa(elementsGPU, nodesGPU, ct, sets);\n');
        
        % Index computation
        fprintf(fileID,'\n%s\n','%% Index GPU');
        fprintf(fileID,'[iK, jK] = Index_spsa(elementsGPU, sets);\n');
        fprintf(fileID,'wait(dev);\n');
        
        % Numerical integration
        fprintf(fileID,'\n%s\n','%% Local ke GPU');
        fprintf(fileID,'Ke = eStiff_spsa(elementsGPU, nodesGPU, ct, sets);\n');
        fprintf(fileID,'wait(dev);\n');
        
        % Assembly
        fprintf(fileID,'\n%s\n','%% Assembly GPU');
        fprintf(fileID,'K = AssemblyStiffMa(iKd, jKd, Ked, sets);\n');
        fprintf(fileID,'wait(dev);\n');
        
        
    elseif strcmp(sets.prob_type, 'Vector')
        fprintf(fileID,'sets.dxn = 3;\n');
        sets.dxn = 3;
        fprintf(fileID,'sets.edof = sets.dxn * sets.nxe;\n');
        fprintf(fileID,'sets.sz = (sets.edof * (sets.edof + 1) )/2;\n');
        fprintf(fileID,'sets.tdofs = sets.nnod * sets.dxn;\n');
        
        % For assembly
        fprintf(fileID,'\n%s\n', '% For GPU assembly');
        fprintf(fileID,'[iKd, jKd] = Index_vpsa(elementsGPU, sets);\n');
        fprintf(fileID,'Ked = eStiff_vpsa(elementsGPU, nodesGPU, MP, sets);\n');
        
        % Index computation
        fprintf(fileID,'\n%s\n','%% Index GPU');
        fprintf(fileID,'[iK, jK] = Index_vpsa(elementsGPU, sets);\n');
        fprintf(fileID,'wait(dev);\n');
        
        % Numerical integration
        fprintf(fileID,'\n%s\n','%% Local ke GPU');
        fprintf(fileID,'Ke = eStiff_vpsa(elementsGPU, nodesGPU, MP, sets);\n');
        fprintf(fileID,'wait(dev);\n');
        
        % Assembly
        fprintf(fileID,'\n%s\n','%% Assembly GPU');
        fprintf(fileID,'K = AssemblyStiffMa(iKd, jKd, Ked, sets);\n');
        fprintf(fileID,'wait(dev);\n');
        
    end
    
    
    
    
    
    
elseif strcmp(sets.proc_type, 'CPU')
    
    if strcmp(sets.prob_type, 'Scalar')
        % For assembly
        fprintf(fileID,'\n%s\n', '% For CPU assembly');
        fprintf(fileID,'[Iar, tnel, tdof] = Index_sosa(nel, nel, nel);\n');
        fprintf(fileID,'Ke = eStiff_sosa(ct, tnel);\n');
        
        % Index computation
        fprintf(fileID,'\n%s\n','%% Index CPU');
        fprintf(fileID,'[Indx, nels, tdofs] = Index_sosa(nel, nel, nel);\n');
        
        % Numerical integration
        fprintf(fileID,'\n%s\n','%% Local ke CPU');
        fprintf(fileID,'Keall = eStiff_sosa(ct, tnel);\n');
        
        % Assembly
        fprintf(fileID,'\n%s\n','%% Assembly CPU');
        fprintf(fileID,'K = AssemblyStiffMa_CPUo(Iar(:,1), Iar(:,2), Ke, tdof);\n');
        
        
    elseif strcmp(sets.prob_type, 'Vector')
        % For assembly
        fprintf(fileID,'\n%s\n', '% For CPU assembly');
        fprintf(fileID,'[Iar, tnel, tdof] = Index_vosa(nel, nel, nel);\n');
        fprintf(fileID,'Ke = eStiff_vosa(MP, tnel);\n');
        
        % Index computation
        fprintf(fileID,'\n%s\n','%% Index CPU');
        fprintf(fileID,'[Indx, nels, tdofs] = Index_vosa(nel, nel, nel);\n');
        
        % Numerical integration
        fprintf(fileID,'\n%s\n','%% Local ke CPU');
        fprintf(fileID,'Keall = eStiff_vosa(MP, tnel);\n');
        
        % Assembly
        fprintf(fileID,'\n%s\n','%% Assembly CPU');
        fprintf(fileID,'K = AssemblyStiffMa_CPUo(Iar(:,1), Iar(:,2), Ke, tdof);\n');
        
    end
    
    
end

fclose(fileID);
