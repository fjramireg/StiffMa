% SCALAR folder contains all the necessary code to assembly global sparse
% stiffness matrices from finite element analysis of scalar problems like
% electrical and thermal phenomena. 
% Version 1.4 13-Dec-2019
% 
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.
% 
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
% 
% 
% Files
% 




%   IndexScalarsap    - Compute the row/column indices of tril(K) in PARALLEL computing
%   IndexScalarsas    - Compute the row/column indices of tril(K) using SERIAL computing

%   Hex8scalars       - Compute the element stiffnes matrix for a SCALAR problem in SERIAL computing.
%   Hex8scalarsap     - Compute all tril(ke) for a SCALAR problem in PARALLEL computing
%   Hex8scalarsas     - Compute the lower symmetric part of all ke in SERIAL computing
%   Hex8scalarss      - Compute the lower symmetric part of the element stiffness matrix

%   StiffMas          - Create the global stiffness matrix K for a SCALAR problem in SERIAL computing.
%   StiffMass         - Create the global stiffness matrix tril(K) for a SCALAR problem in SERIAL computing
%   StiffMaps         - Create the global stiffness matrix tril(K) for a SCALAR problem in PARALLEL computing

%   runScalarCPUvsGPU - Script to run the whole assembly code on the CPU and GPU, and compare them
%   runScalarOnCPU    - run the whole assembly code on the CPU
%   runScalarOnGPU    - run the whole assembly code on the GPU
%   runIndexCPUvsGPU  - Script to run the INDEX code on the CPU and GPU, and compare them
%   runIndexOnCPU     - Script to run the INDEX code on CPU
%   runIndexOnGPU     - Script to run the INDEX code on GPU
%   runHex8CPUvsGPU   - Script to run the HEX8 (ke) code on the CPU and GPU, and compare them
%   runHex8OnCPU      - Script to run the HEX8 (ke) code on the CPU
%   runHex8OnGPU      - Script to run the HEX8 (ke) code on the GPU

