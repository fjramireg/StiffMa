% SCALAR folder contains all the necessary code to assembly global sparse
% stiffness matrices from finite element analysis of scalar problems like
% electrical and thermal phenomena. 
% Version 1.4 07-Dec-2019
% 
% Files


%   runmeScalarOnCPU - run the whole assembly code on the CPU
%   runmeScalarOnGPU - run the whole assembly code on the GPU

%   IndexScalarsap   - Compute the row/column indices of tril(K) in PARALLEL computing
%   IndexScalarsas   - Compute the row/column indices of tril(K) using SERIAL computing

%   Hex8scalars      - Compute the element stiffnes matrix for a SCALAR problem in SERIAL computing
%   Hex8scalarsap    - Compute all tril(ke) for a SCALAR problem in PARALLEL computing
%   Hex8scalarsas    - Compute the lower symmetric part of all ke in SERIAL computing
%   Hex8scalarss     - Compute the lower symmetric part of the element stiffness matrix in SERIAL computing

%   StiffMas         - Create the global stiffness matrix K for a SCALAR problem in SERIAL computing.
%   StiffMass        - Create the global stiffness matrix tril(K) for a SCALAR problem in SERIAL computing
%   StiffMaps        - Create the global stiffness matrix tril(K) for a SCALAR problem in PARALLEL computing
