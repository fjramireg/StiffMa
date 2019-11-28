function WrittingPerfFnSpvsAcc(nel,dTE,dTN,proctype,prob_type)
% Writes a script to the performance test of "sparse" vs "accumarray"

fID = fopen('perftestSpvsAcc.m','w');

if strcmp(prob_type,'Scalar')
    
    if strcmp(proctype,'CPUp')
        
        fprintf(fID,'nel = %d;\n',nel);
        fprintf(fID,"dTE = '%s';\n",dTE);
        fprintf(fID,"dTN = '%s';\n",dTN);
        fprintf(fID,'[elements, nodes] = CreateMesh(nel,nel,nel,dTE,dTN,0,0);\n');
        fprintf(fID,'c = 1;\n');
        fprintf(fID,'N = size(nodes,1);\n');
        fprintf(fID,'[iK, jK] = IndexScalarSymCPUp(elements);\n');
        fprintf(fID,'Ke = Hex8scalarSymCPUp(elements,nodes,c);\n\n');
        fprintf(fID,'%s\n','%% Assembly of K (scalar) using sparse on CPU');
        fprintf(fID,'K = sparse(iK, jK, Ke, N, N);\n\n');
        fprintf(fID,'%s\n','%% Assembly of K (scalar) using accumarray on CPU');
        fprintf(fID,'K = accumarray([iK,jK], Ke, [N,N], [], [], 1);\n');
        
    elseif strcmp(proctype,'GPU')
        
        fprintf(fID,'nel = %d;\n',nel);
        fprintf(fID,"dTE = '%s';\n",dTE);
        fprintf(fID,"dTN = '%s';\n",dTN);
        fprintf(fID,'[elements, nodes] = CreateMesh(nel,nel,nel,dTE,dTN,0,0);\n');
        fprintf(fID,'c = 1;\n');
        fprintf(fID,'N = size(nodes,1);\n');
        fprintf(fID,"[iK, jK] = IndexScalarSymGPU(gpuArray(elements'));\n");
        fprintf(fID,"Ke = Hex8scalarSymGPU(gpuArray(elements'),gpuArray(nodes'),c);\n\n");
        fprintf(fID,'%s\n','%% Assembly of K (scalar) using sparse on GPU');
        fprintf(fID,'K = sparse(iK, jK, Ke, N, N);\n\n');
        fprintf(fID,'%s\n','%% Assembly of K (scalar) using accumarray on GPU');
        fprintf(fID,'K = accumarray([iK,jK], Ke, [N,N], [], [], 1);\n');
        fprintf(fID,'%s\n','%% Reset GPU device');
        fprintf(fID,'reset(gpuDevice);');
        
    else
        error('Error. No processor type defined.');
    end
    
elseif strcmp(prob_type,'Vector')
    
    if strcmp(proctype,'CPUp')
        
        fprintf(fID,'nel = %d;\n',nel);
        fprintf(fID,"dTE = '%s';\n",dTE);
        fprintf(fID,"dTN = '%s';\n",dTN);
        fprintf(fID,'[elements, nodes] = CreateMesh(nel,nel,nel,dTE,dTN,0,0);\n');
        fprintf(fID,'E = 200e9;\n');
        fprintf(fID,'nu = 0.3;\n');
        fprintf(fID,'N = size(nodes,1);\n');
        fprintf(fID,'[iK, jK] = IndexVectorSymCPUp(elements);\n');
        fprintf(fID,'Ke = Hex8vectorSymCPUp(elements,nodes,E,nu);\n\n');
        fprintf(fID,'%s\n','%% Assembly of K (vector) using sparse on CPU');
        fprintf(fID,'K = sparse(iK, jK, Ke, 3*N, 3*N);\n\n');
        fprintf(fID,'%s\n','%% Assembly of K (vector) using accumarray on CPU');
        fprintf(fID,'K = accumarray([iK,jK], Ke, [3*N,3*N], [], [], 1);\n');
        
    elseif strcmp(proctype,'GPU')
        
        fprintf(fID,'nel = %d;\n',nel);
        fprintf(fID,"dTE = '%s';\n",dTE);
        fprintf(fID,"dTN = '%s';\n",dTN);
        fprintf(fID,'[elements, nodes] = CreateMesh(nel,nel,nel,dTE,dTN,0,0);\n');
        fprintf(fID,'E = 200e9;\n');
        fprintf(fID,'nu = 0.3;\n');
        fprintf(fID,'N = size(nodes,1);\n');
        fprintf(fID,"[iK, jK] = IndexVectorSymGPU(gpuArray(elements'));\n");
        fprintf(fID,"Ke = Hex8vectorSymGPU(gpuArray(elements'),gpuArray(nodes'),E,nu);\n\n");
        fprintf(fID,'%s\n','%% Assembly of K (vector) using sparse on GPU');
        fprintf(fID,'K = sparse(iK, jK, Ke, 3*N, 3*N);\n\n');
        fprintf(fID,'%s\n','%% Assembly of K (vector) using accumarray on GPU');
        fprintf(fID,'K = accumarray([iK,jK], Ke, [3*N,3*N], [], [], 1);\n');
        fprintf(fID,'%s\n','%% Reset GPU device');
        fprintf(fID,'reset(gpuDevice);');
        
    else
        error('Error. No processor type defined.');
    end
    
else
    error('Error. No problem type defined.');
end

fclose(fID);
