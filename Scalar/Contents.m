% SCALAR folder contains all the necessary code to assembly global sparse
% stiffness matrices from finite element analysis of scalar problems like
% electrical and thermal phenomena. 
% Version 1.3 21-Jan-2019
% 
% Files
%   Hex8scalar          - Compute the element stiffness matrix for a SCALAR problem.
%   Hex8scalarSym       - Compute the lower symmetric part of the element stiffness
%   Hex8scalarSymGPU    - Compute the lower symmetric part of all the element
%   IndexScalarSymGPU   - Compute the row and column indices of lower symmetric
%   runmeScalar         - run the whole assembly code on the CPU and GPU
%   Hex8scalarSymCPU    - Compute the lower symmetric part of all the element
%   IndexScalarSymCPU   - Compute the row and column indices of lower symmetric
%   StiffMatGenSc       - Create the global stiffness matrix for a SCALAR problem.
%   StiffMatGenScSym    - Create the global stiffness matrix for a SCALAR problem
%   StiffMatGenScSymGPU - Create the global stiffness matrix for a SCALAR

