% VECTOR folder contains all the necessary code to assembly global sparse
% stiffness matrices from finite element analysis of vector problems like
% structural phenomena.
% Version 1.4 31-Jan-2020

%   The name convention: After underscore (_) letters mean:
%       (s,v): Refers to scalar (s) or vector (v) problem
%       (s,p): Refers to serial (s) or parallel (p) computing
%       (s): For the symmetry case only


%% Functions Files
%   DMatrix                - Compute the isotropic material matrix for the VECTOR problem.
%   Index_vps              - Compute the row/column indices of tril(K) in a vector (v)
%   Index_vss              - Compute the row/column indices of tril(K) in a vector (v)
%   StiffMa_vps            - Create the global stiffness matrix for a VECTOR (v) problem
%   StiffMa_vs             - Create the global stiffness matrix K for a VECTOR (v) problem
%   StiffMa_vss            - Create the global stiffness matrix for a VECTOR (v) problem
%   eStiff_vs              - Compute the element stiffness matrix for a VECTOR (s) problem
%   eStiff_vss             - Compute the element stiffness matrix for a VECTOR (v) problem in
%   eStiffa_vps            - Compute ALL (a) element stiffness matrices for a VECTOR (v)
%   eStiffa_vs             - Compute ALL (a) the element stiffness matrices for a VECTOR (v)
%   eStiffa_vss            - Compute ALL (a) the element stiffness matrices for a VECTOR (v)

%% Script Files used to run the functions
%   runHex8VectorCPUvsGPU  - Script to run the HEX8 (ke) code. CPU vs GPU
%   runHex8VectorOnCPU     - Script to run the HEX8 (ke) code on the CPU
%   runHex8VectorOnGPU     - Script to run the HEX8 (ke) code on the GPU
%   runIndexVectorCPUvsGPU - Runs the INDEX vector code. CPU vs GPU
%   runIndexVectorOnCPU    - Runs the INDEX vector code on the CPU
%   runIndexVectorOnGPU    - Runs the INDEX vector code on the GPU
%   runVectorCPUvsGPU      - Runs the whole assembly vector code on the CPU and GPU (comparison)
%   runVectorOnCPU         - Runs the whole assembly vector code on the CPU
%   runVectorOnGPU         - Runs the whole assembly vector code on the GPU

