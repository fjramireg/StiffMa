% VECTOR folder contains all the necessary code to assembly global sparse
% stiffness matrices from finite element analysis of vector problems like
% structural phenomena. 
% Version 1.3 21-Jan-2019
% 
% Files
%   Hex8vector           - Compute the element stiffness matrix for a VECTOR problem.
%   Hex8vectorSym        - Compute the lower symmetric part of the element stiffness
%   Hex8vectorSymGPU     - Compute the lower symmetric part of the element stiffness
%   IndexVectorSymGPU    - Compute the row and column indices of lower symmetric
%   MaterialMatrix       - Compute the isotropic material matrix for the VECTOR problem
%   runmeVector          - run the whole assembly code on the CPU and GPU
%   Hex8vectorSymCPU     - Compute the lower symmetric part of all the element
%   Hex8vectorSymCPUp    - Compute the lower symmetric part of all the element
%   IndexVectorSymCPU    - Compute the row and column indices of lower symmetric
%   IndexVectorSymCPUp   - Compute the row and column indices of lower symmetric
%   StiffMatGenVc        - Create the global stiffness matrix for a VECTOR problem.
%   StiffMatGenVcSymCPU  - Create the global stiffness matrix for a VECTOR
%   StiffMatGenVcSymCPUp - Create the global stiffness matrix for a VECTOR
%   StiffMatGenVcSymGPU  - Create the global stiffness matrix for a VECTOR


