function ndiv = WriteStiffMaPerfScript2(sets)
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
fprintf(fileID,"MP.c = 384.1;\n");
fprintf(fileID,"MP.E = 200e9;\n");
fprintf(fileID,"MP.nu = 0.3;\n");

% Mesh generation
fprintf(fileID,'\n%s\n','% Mesh generation');
fprintf(fileID,'[Mesh.elements, Mesh.nodes] = CreateMesh2(nel, nel, nel, sets.dTE, sets.dTN);\n');
fprintf(fileID,'[sets.nel, sets.nxe]  = size(Mesh.elements);\n');
fprintf(fileID,'[sets.nnod, sets.dim] = size(Mesh.nodes);\n');
if strcmp(sets.prob_type,'Scalar')
    fprintf(fileID,'sets.dxn = 1;\n');
    sets.dxn = 1;
elseif strcmp(sets.prob_type,'Vector')
    fprintf(fileID,'sets.dxn = 3;\n');
    sets.dxn = 3;
else
    error('Problem not defined!');
end
fprintf(fileID,'sets.edof = sets.dxn * sets.nxe;\n');
fprintf(fileID,'sets.sz = (sets.edof * (sets.edof + 1) )/2;\n');
fprintf(fileID,'sets.tdofs = sets.nnod * sets.dxn;\n');
% 
[Mesh.elements, Mesh.nodes] = CreateMesh2(sets.nel, sets.nel, sets.nel, sets.dTE, sets.dTN);
[sets.nel, sets.nxe]  = size(Mesh.elements);
[sets.nnod, sets.dim] = size(Mesh.nodes);
sets.edof = sets.dxn * sets.nxe;
sets.sz = (sets.edof * (sets.edof + 1) )/2;
sets.tdofs = sets.nnod * sets.dxn;

% GPU setup
fprintf(fileID,'\n%s\n', '% GPU setup');
fprintf(fileID,'dev = gpuDevice;\n');
fprintf(fileID,'sets.tbs = dev.MaxThreadsPerBlock;\n');
fprintf(fileID,'sets.numSMs   = dev.MultiprocessorCount;\n');
fprintf(fileID,'sets.WarpSize = dev.SIMDWidth;\n');

% Determination of number of chunks based on current GPU memory
fprintf(fileID,'\n%s\n', '% Number of chuncks');
fprintf(fileID,'d_et  = zeros(1,1,sets.dTE);\n');
fprintf(fileID,"d_et1 = whos('d_et');\n");
fprintf(fileID,'szInd = d_et1.bytes;\n');
fprintf(fileID,'d_nt  = zeros(1,1,sets.dTN);\n');
fprintf(fileID,"d_nt1 = whos('d_nt');\n");
fprintf(fileID,'szNNZ = d_nt1.bytes;\n');
fprintf(fileID,'Mmesh  = szInd*numel(Mesh.elements) + szNNZ*numel(Mesh.nodes);\n');
fprintf(fileID,'Mtrip  = (2*szInd + szNNZ)*sets.sz*sets.nel;\n');
fprintf(fileID,'Maccum = 3*Mtrip;\n');
fprintf(fileID,'Mcsc   = 0.5*Mtrip;\n');
fprintf(fileID,'Mtotal = Mmesh + Mtrip + Maccum + Mcsc;\n');
fprintf(fileID,'ndiv = ceil(Mtotal/dev.AvailableMemory);\n');
% fprintf(fileID,'ndiv = ndiv + sets.sf*(ndiv>1);\n');
fprintf(fileID,'ndiv = ndiv + sets.sf;\n');
fprintf(fileID,'while mod(sets.nel,ndiv) ~= 0\n');
fprintf(fileID,'    ndiv = ndiv + 1;\n');
fprintf(fileID,'end\n');
% Number of chuncks
d_et  = zeros(1,1,sets.dTE);%#ok
d_et1 = whos('d_et');
szInd = d_et1.bytes;
d_nt  = zeros(1,1,sets.dTN);%#ok
d_nt1 = whos('d_nt');
szNNZ = d_nt1.bytes;
Mmesh  = szInd*numel(Mesh.elements) + szNNZ*numel(Mesh.nodes);
Mtrip  = (2*szInd + szNNZ)*sets.sz*sets.nel;
Maccum = 3*Mtrip;
Mcsc   = 0.5*Mtrip;
Mtotal = Mmesh + Mtrip + Maccum + Mcsc;
dev = gpuDevice;
ndiv = ceil(Mtotal/dev.AvailableMemory);
% ndiv = ndiv + sets.sf*(ndiv>1);
ndiv = ndiv + sets.sf;
while mod(sets.nel,ndiv) ~= 0
    ndiv = ndiv + 1;
end

% Available memory check
fprintf(fileID,'\n%s\n', '% Available memory check');
fprintf(fileID,'Mtotal_c = Mmesh + Mtrip*(4/ndiv + 1/2);\n');
fprintf(fileID,'if Mtotal_c > dev.AvailableMemory\n');
fprintf(fileID,'    reset(dev);\n');
fprintf(fileID,"    x =['No enough memory on the GPU to process the mesh with ', num2str(ndiv), ' chunks'];\n");
fprintf(fileID,"    error(x);\n");
fprintf(fileID,'else\n');
fprintf(fileID,"    x=['The global stiffness matrix will be computed with ', num2str(ndiv), ' chunk(s).'];\n");
fprintf(fileID,"    disp(x);\n");
fprintf(fileID,'end\n');

% Transfer memory: host to device
fprintf(fileID,'\n%s\n', '% Transfer memory: host to device');
fprintf(fileID,"x=['Available memory on GPU before computations begin (MB): ', num2str(dev.AvailableMemory/1e6)];\n");
fprintf(fileID,"disp(x);\n");
fprintf(fileID,"elementsGPU = gpuArray(Mesh.elements');\n");
fprintf(fileID,"nodesGPU = gpuArray(Mesh.nodes');\n");

% 'Scalar'
if strcmp(sets.prob_type,'Scalar')
    
    fprintf(fileID,"x=['Processing the SCALAR problem with ', num2str(nel),'x',num2str(nel),'x',num2str(nel), ' elements'];\n");
    fprintf(fileID,"disp(x);\n");
    
    % Partitioning
    fprintf(fileID,'\n%s\n', '%% StiffMa');
    fprintf(fileID,"if (ndiv > 1)\n");
    fprintf(fileID,"    m = sets.nel / ndiv;\n");
    fprintf(fileID,"    sets.nel = m;\n");
    fprintf(fileID,"    K = sparse(sets.tdofs, sets.tdofs);\n");
    fprintf(fileID,"    for i=1:ndiv\n");
%     fprintf(fileID,"        x =['\t Processing Chunk ', num2str(i), ' of ', num2str(ndiv), '...'];\n");
%     fprintf(fileID,"        disp(x);\n");
%     fprintf(fileID,"        disp('.');\n");
    fprintf(fileID,"        ini = 1 + m*(i-1);\n");
    fprintf(fileID,"        fin = ini + m - 1;\n");
    fprintf(fileID,"        [iKd, jKd] = Index_spsa(elementsGPU(:, ini:fin), sets);\n");
    fprintf(fileID,"        Ked = eStiff_spsa(elementsGPU(:, ini:fin), nodesGPU, MP.c, sets);\n");
    fprintf(fileID,"        wait(dev);\n");
    fprintf(fileID,"        K = K + AssemblyStiffMa(iKd, jKd, Ked, sets);\n");
    fprintf(fileID,"    end\n");
    fprintf(fileID,"    clear elementsGPU nodesGPU iKd jKd Ked\n");
    % Without Partitioning
    fprintf(fileID,"else\n");
    fprintf(fileID,"    [iKd, jKd] = Index_spsa(elementsGPU, sets);\n");
    fprintf(fileID,"    Ked = eStiff_spsa(elementsGPU, nodesGPU, MP.c, sets);\n");
    fprintf(fileID,"    wait(dev);\n");
    fprintf(fileID,"    clear elementsGPU nodesGPU\n");
    fprintf(fileID,"    K = AssemblyStiffMa(iKd, jKd, Ked, sets);\n");
    fprintf(fileID,"    clear iKd jKd Ked\n");
    fprintf(fileID,"end\n");
    fprintf(fileID,"wait(dev);\n");
    
    % 'Vector'
elseif strcmp(sets.prob_type,'Vector')
    
    fprintf(fileID,"x=['Processing the VECTOR problem with ', num2str(nel),'x',num2str(nel),'x',num2str(nel), ' elements'];\n");
    fprintf(fileID,"disp(x);\n");
    
    % Partitioning
    fprintf(fileID,'\n%s\n', '%% StiffMa');
    fprintf(fileID,"if (ndiv > 1)\n");
    fprintf(fileID,"    m = sets.nel / ndiv;\n");
    fprintf(fileID,"    sets.nel = m;\n");
    fprintf(fileID,"    K = sparse(sets.tdofs, sets.tdofs);\n");
    fprintf(fileID,"    for i=1:ndiv\n");
%     fprintf(fileID,"        x =['\t Processing Chunk ', num2str(i), ' of ', num2str(ndiv), '...'];\n");
%     fprintf(fileID,"        disp(x);\n");
%     fprintf(fileID,"        disp('.');\n");
    fprintf(fileID,"        ini = 1 + m*(i-1);\n");
    fprintf(fileID,"        fin = ini + m - 1;\n");
    fprintf(fileID,"        [iKd, jKd] = Index_vpsa(elementsGPU(:, ini:fin), sets);\n");
    fprintf(fileID,"        Ked = eStiff_vpsa(elementsGPU(:, ini:fin), nodesGPU, MP, sets);\n");
    fprintf(fileID,"        wait(dev);\n");
    fprintf(fileID,"        K = K + AssemblyStiffMa(iKd, jKd, Ked, sets);\n");
    fprintf(fileID,"    end\n");
    fprintf(fileID,"    clear elementsGPU nodesGPU iKd jKd Ked\n");
    % Without Partitioning
    fprintf(fileID,"else\n");
    fprintf(fileID,"    [iKd, jKd] = Index_vpsa(elementsGPU, sets);\n");
    fprintf(fileID,"    Ked = eStiff_vpsa(elementsGPU, nodesGPU, MP, sets);\n");
    fprintf(fileID,"    wait(dev);\n");
    fprintf(fileID,"    clear elementsGPU nodesGPU\n");
    fprintf(fileID,"    K = AssemblyStiffMa(iKd, jKd, Ked, sets);\n");
    fprintf(fileID,"    clear iKd jKd Ked\n");
    fprintf(fileID,"end\n");
    fprintf(fileID,"wait(dev);\n");
    
else
    error('Error. No problem type defined.');
end

fclose(fileID);
