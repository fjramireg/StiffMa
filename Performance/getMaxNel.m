function nel_max = getMaxNel(dTE,dTN,Problem,code_version,safetyFactor)
% Return the maximum number of elements that will fit in the device memory
%
%   getTestSizes('int32','double','Scalar','GPU',2)

% Size of...
SizeOfEData = sizeof( dTE );        % Size of elemental data type
SizeOfNData = sizeof( dTN );        % Size of nodal data type

% Number of entries that depends on the problem type
if strcmp(Problem,'Scalar')
    SizeOfProb = 36;
elseif strcmp(Problem,'Vector')
    SizeOfProb = 300;
else
    error('No problem type defined!');
end

% Memory based on the code version
if strcmp(code_version, 'CPUs')  	% CPU serial code
    safetyFactor = safetyFactor*2;  % On the host everything takes longer, so don't go as far
    freeMem = 4*2^30;               % If no GPU to get memory size, so just go for 4GB
elseif strcmp(code_version, 'CPUp') % CPU parallel code
    freeMem = 96*2^30;              % If no GPU to get memory size, so just go for 16GB
elseif strcmp(code_version, 'GPU')  % GPU parallel code
    gpu = gpuDevice();
    freeMem = gpu.AvailableMemory;  % Use as much memory as we can
    % freeMem = gpu.FreeMemory;     % What's the difference?
end

% Definition of maximum mesh size
reqMem = 8*SizeOfEData ...          % Memory requirement to store "elements" array
    + 3*SizeOfNData ....            % To store "nodes" array (approx.)
    + 3*SizeOfProb*SizeOfEData ...  % To store "iK" and "jK" arrays & [i,j]=find(K)
    + 2*SizeOfProb*SizeOfNData;     % To store "Ke" array & [~,~,v]=find(K)
Nelmax = freeMem / (reqMem*safetyFactor) ;
nel_max = floor(Nelmax^(1/3)/10)*10;
