function reqMem = RequiredMemory(nel,nnod,FunctionType)
% Computes the required memory depending on the mesh size and function type

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

% Type of function to be computed
if strcmp(FunctionType,'Index')
    reqMem = 8*nel*SizeOfEData ...      % To store "elements" array
        + 2*SizeOfProb*nel*SizeOfEData; % To store "iK" and "jK" arrays
elseif strcmp(FunctionType,'ElementStiffness')
    reqMem = 8*nel*SizeOfEData ...      % To store "elements" array
        + 3*nnod*SizeOfNData ....       % To store "nodes" array
        + SizeOfProb*nel*SizeOfNData;   % To store "Ke" array
elseif (strcmp(FunctionType,'GlobalStiffness') || strcmp(FunctionType,'All'))
    reqMem = 8*nel*SizeOfEData ...      % To store "elements" array
        + 3*nnod*SizeOfNData ...        % To store "nodes" array
        + 2*SizeOfProb*nel*SizeOfEData...% To store "iK" and "jK" arrays
        + SizeOfProb*nel*SizeOfNData;   % To store "Ke" array
else
    error('No function type defined!');
end
