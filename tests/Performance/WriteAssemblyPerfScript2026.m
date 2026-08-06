function WriteAssemblyPerfScript2026(sets)
% Writes a script to measure the performance of the code using "runperf"

%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.
%
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Institución Universitaria Pascual Bravo, Medellin-Colombia
%       Updated: August 03, 2026.
%       Created:  12/02/2020. Version: 1.4

% Validate input 'sets' minimally to provide clearer errors early.
if nargin < 1 || ~isstruct(sets)
    error('WriteAssemblyPerfScript2026 requires a structure input ''sets''.');
end
reqFields = {'name','nel','dTE','dTN','proc_type','prob_type'};
missing = setdiff(reqFields, fieldnames(sets));
if ~isempty(missing)
    error('Input ''sets'' is missing required fields: %s', strjoin(missing,', '));
end

Filename = [sets.name,'.m'];
fileID = fopen(Filename,'w');
fprintf(fileID,'sets.nel = %d;\n',sets.nel);
fprintf(fileID,"sets.dTE = '%s';\n",sets.dTE);
fprintf(fileID,"sets.dTN = '%s';\n",sets.dTN);
fprintf(fileID,'[Mesh.elements, Mesh.nodes] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets);\n');
fprintf(fileID,'sets.nel = %d;\n',sets.nel^3);
fprintf(fileID,'sets.nnodes = %d;\n',(sets.nel+1)^3);
testname = ['%% Assembly_',sets.proc_type,'_',sets.prob_type,'_',sets.dTN,'_',num2str(sets.nel)];

% 'Scalar'
if strcmp(sets.prob_type,'Scalar')
    fprintf(fileID,"sets.sz = %d;\n",36);
    fprintf(fileID,"sets.edof = %d;\n",8);
    fprintf(fileID,"sets.tdofs = sets.nnodes * 1;\n");
    fprintf(fileID,"c = %d;\n",384.1);
    fprintf(fileID,"d = gpuDevice;\n");
    fprintf(fileID,"sets.tbs = d.MaxThreadsPerBlock;\n");
    fprintf(fileID,"sets.numSMs   = d.MultiprocessorCount;\n");
    fprintf(fileID,"sets.WarpSize = d.SIMDWidth;\n");    
    fprintf(fileID,'[iK, jK] = Index_spsa_opt(gpuArray(Mesh.elements), sets);\n');
    fprintf(fileID,'Ke = eStiff_spsa_opt(gpuArray(Mesh.elements), gpuArray(Mesh.nodes), c, sets);\n');
    fprintf(fileID,'wait(d);\n');
    
    % 'Scalar'-'CPU'
    if strcmp(sets.proc_type,'CPU')
        
        % 'Scalar'-'CPU'-'Symmetry'
        fprintf(fileID,"[iK, jK, Ke] = gather(iK, jK, Ke);\n");        
        fprintf(fileID,'\n%s\n',testname);
        fprintf(fileID,'K = AssemblyStiffMa(iK, jK, Ke, sets);\n');
        
        % 'Scalar'-'GPU'-'Symmetry'
    elseif strcmp(sets.proc_type,'GPU')
        fprintf(fileID,'\n%s\n',testname);
        fprintf(fileID,'K = AssemblyStiffMa(iK, jK, Ke, sets);\n');
        fprintf(fileID,'wait(d);\n');
        
    else
        error('Error. No processor type defined.');
    end    
    
    % 'Vector'
elseif strcmp(sets.prob_type,'Vector')
    fprintf(fileID,"sets.sz = %d;\n",300);
    fprintf(fileID,"sets.edof = %d;\n",24);
    fprintf(fileID,"sets.tdofs = sets.nnodes * 3;\n");
    fprintf(fileID,"MP.E = %d;\n",200e9);
    fprintf(fileID,"MP.nu = %d;\n",0.3);
    fprintf(fileID,"d = gpuDevice;\n");
    fprintf(fileID,"sets.tbs = d.MaxThreadsPerBlock;\n");
    fprintf(fileID,"sets.numSMs   = d.MultiprocessorCount;\n");
    fprintf(fileID,"sets.WarpSize = d.SIMDWidth;\n");   
    fprintf(fileID,'[iK, jK] = Index_vpsa_opt(gpuArray(Mesh.elements), sets);\n');
    fprintf(fileID,'Ke = eStiff_vpsa_opt(gpuArray(Mesh.elements), gpuArray(Mesh.nodes), MP, sets);\n');
    fprintf(fileID,'wait(d);\n');
    
    % 'Vector'-'CPU'
    if strcmp(sets.proc_type,'CPU')
                
        % 'Vector'-'CPU'-'Symmetry'
        fprintf(fileID,"[iK, jK, Ke] = gather(iK, jK, Ke);\n");        
        fprintf(fileID,'\n%s\n',testname);
        fprintf(fileID,'K = AssemblyStiffMa(iK, jK, Ke, sets);\n');
        
        % 'Vector'-'GPU'-'Symmetry'
    elseif strcmp(sets.proc_type,'GPU')
        fprintf(fileID,'\n%s\n',testname);
        fprintf(fileID,'K = AssemblyStiffMa(iK, jK, Ke, sets);\n');
        fprintf(fileID,'wait(d);\n');
        
    else
        error('Error. No processor type defined.');
    end
    
else
    error('Error. No problem type defined.');
end

fclose(fileID);
