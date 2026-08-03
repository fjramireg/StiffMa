function WriteIndexPerfScript2026(sets)
% WriteIndexPerfScript2026 Generates a performance test script for indexing routines
%
% WriteIndexPerfScript2026(sets) creates a MATLAB script file named
% <sets.name>.m that sets up the variable 'sets', generates a mesh, and
% writes calls to the appropriate indexing functions for benchmarking with
% runperf. The generated script configures problem-dependent parameters
% (Scalar or Vector), data precision type (uint32 or uint64) and
% processor-dependent settings (CPU or GPU).
%
% Input:
%   sets - structure with required fields:
%       name       : base name for output script (string)
%       nel        : number of elements per spatial direction (scalar)
%       dTE        : descriptor for element type (string)
%       dTN        : descriptor for node type (string)
%       proc_type  : 'CPU' or 'GPU' (string)
%       prob_type  : 'Scalar' or 'Vector' (string)
%
% The produced file contains lines that:
%   - set sets.nel and other parameters,
%   - call CreateMesh2 to build 'elements',
%   - and call the appropriate Index_* routine for the chosen problem
%     and processor type.
%
% Example:
%   s.name = 'MyPerfTest';
%   s.nel = 8;
%   s.dTE = 'uint32';
%   s.dTN = 'single';
%   s.proc_type = 'GPU';
%   s.prob_type = 'Scalar';
%   WriteIndexPerfScript2026(s);
%
% Notes:
%   - The function assumes CreateMesh2 and the Index_* functions are on the
%     MATLAB path.
%   - For GPU scripts the generated file queries gpuDevice properties and
%     transfers 'elements' to the GPU using gpuArray.
%   - The function overwrites an existing file named <sets.name>.m.
%
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.
%
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Institución Universitaria Pascual Bravo, Medellin-Colombia
%       Updated: August 03, 2026.
%       Created:  12/02/2020. Version: 1.4

% Validate input 'sets' minimally to provide clearer errors early.
if nargin < 1 || ~isstruct(sets)
    error('WriteIndexPerfScript2026 requires a structure input ''sets''.');
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
fprintf(fileID,'[elements, ~] = CreateMesh2(sets.nel,sets.nel,sets.nel,sets);\n');
fprintf(fileID,'sets.nel = %d;\n',sets.nel^3);
testname = ['%% Index_',sets.proc_type,'_',sets.prob_type,'_',sets.dTE,'_',num2str(sets.nel)];

%% 'Scalar'
if strcmp(sets.prob_type,'Scalar')
    fprintf(fileID,"sets.sz = %d;\n",36);
    fprintf(fileID,"sets.edof = %d;\n",8);

    % 'Scalar'-'CPU'
    if strcmp(sets.proc_type,'CPU')

        % 'Scalar'-'CPU'-'Symmetry'
        fprintf(fileID,'\n%s\n',testname);
        fprintf(fileID,'[iK, jK] = Index_sssa(elements, sets);\n');

        % 'Scalar'-'GPU'-'Symmetry'
    elseif strcmp(sets.proc_type,'GPU')
        fprintf(fileID,"d = gpuDevice;\n");
        fprintf(fileID,"sets.tbs = d.MaxThreadsPerBlock;\n");
        fprintf(fileID,"sets.numSMs   = d.MultiprocessorCount;\n");
        fprintf(fileID,"sets.WarpSize = d.SIMDWidth;\n");
        fprintf(fileID,"elementsGPU = gpuArray(elements);\n");
        fprintf(fileID,'\n%s\n',testname);
        fprintf(fileID,'[iKd, jKd] = Index_spsa_opt(elementsGPU, sets);\n');
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

        % 'Vector'-'CPU'-'Symmetry'
        fprintf(fileID,'\n%s\n',testname);
        fprintf(fileID,'[iK, jK] = Index_vssa(elements, sets);\n');

        % 'Vector'-'GPU'-'Symmetry'
    elseif strcmp(sets.proc_type,'GPU')
        fprintf(fileID,"d = gpuDevice;\n");
        fprintf(fileID,"sets.tbs = d.MaxThreadsPerBlock;\n");
        fprintf(fileID,"sets.numSMs   = d.MultiprocessorCount;\n");
        fprintf(fileID,"sets.WarpSize = d.SIMDWidth;\n");
        fprintf(fileID,"elementsGPU = gpuArray(elements);\n");
        fprintf(fileID,'\n%s\n',testname);
        fprintf(fileID,'[iKd, jKd] = Index_vpsa_opt(elementsGPU, sets);\n');
        fprintf(fileID,'wait(d);\n');

    else
        error('Error. No processor type defined.');
    end

else
    error('Error. No problem type defined.');
end

fclose(fileID);
