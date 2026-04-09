function WriteStiffMaPerfScriptCPUvsGPU_v2(sets)
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
        
        % Full assembly
        fprintf(fileID,'\n%s\n', '%% Full GPU assembly');
        fprintf(fileID,'[iKd, jKd] = Index_spsa(elementsGPU, sets);\n');
        fprintf(fileID,'Ked = eStiff_spsa(elementsGPU, nodesGPU, ct, sets);\n');
        fprintf(fileID,'wait(dev);\n');
        fprintf(fileID,'K = AssemblyStiffMa(iKd, jKd, Ked, sets);\n');
        fprintf(fileID,'wait(dev);\n');
        
        
    elseif strcmp(sets.prob_type, 'Vector')
        fprintf(fileID,'sets.dxn = 3;\n');
        sets.dxn = 3;
        fprintf(fileID,'sets.edof = sets.dxn * sets.nxe;\n');
        fprintf(fileID,'sets.sz = (sets.edof * (sets.edof + 1) )/2;\n');
        fprintf(fileID,'sets.tdofs = sets.nnod * sets.dxn;\n');
        
        % Full assembly
        fprintf(fileID,'\n%s\n', '%% Full GPU assembly');
        fprintf(fileID,'[iKd, jKd] = Index_vpsa(elementsGPU, sets);\n');
        fprintf(fileID,'Ked = eStiff_vpsa(elementsGPU, nodesGPU, MP, sets);\n');
        fprintf(fileID,'wait(dev);\n');
        fprintf(fileID,'K = AssemblyStiffMa(iKd, jKd, Ked, sets);\n');
        fprintf(fileID,'wait(dev);\n');
        
    end
    
    
    
    
    
    
elseif strcmp(sets.proc_type, 'CPU')
    
    % Mesh generation
    fprintf(fileID,'\n%s\n','% Mesh generation');
    fprintf(fileID,'[elements, ~] = CreateMesh2(nel, nel, nel, sets);\n');
    
    if strcmp(sets.prob_type, 'Scalar')
        fprintf(fileID,'opts.symmetric = 1;\n');
        fprintf(fileID,'opts.n_node_dof = 1;\n');
        % Full assembly
        fprintf(fileID,'\n%s\n', '%% Full CPU assembly');
        fprintf(fileID,'Ke = eStiff_sosa2(ct);\n');
        fprintf(fileID,"K = AssemblyStiffMa_CPUo2(elements', Ke, opts);\n");
        
        
    elseif strcmp(sets.prob_type, 'Vector')
        fprintf(fileID,'opts.symmetric = 1;\n');
        fprintf(fileID,'opts.n_node_dof = 3;\n');
        % Full assembly
        fprintf(fileID,'\n%s\n', '%% Full CPU assembly');
        fprintf(fileID,'Ke = eStiff_vosa2(MP);\n');
        fprintf(fileID,"K = AssemblyStiffMa_CPUo2(elements', Ke, opts);\n");
        
    end
    
    
end

fclose(fileID);
