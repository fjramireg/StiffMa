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

%   The name convention: After underscore (_) letters mean:
%       (s,v): Refers to scalar (s) or vector (v) problem
%       (s,p): Refers to serial (s) or parallel (p) computing
%       (s): Only if take advantange of symmetry
%       (s): Only if "single" data precision is supported
%		(a): Only if computes ALL elements data

%% Functions Files
%   Index_sps              - INDEXSCALARSAP Compute the row/column indices of tril(K) in PARALLEL computing
%   Index_sss              - INDEXSCALARSAS Compute the row/column indices of tril(K) using SERIAL computing
%   StiffMa_sps            - STIFFMAPS Create the global stiffness matrix tril(K) for a SCALAR problem in PARALLEL computing
%   StiffMa_ss             - STIFFMAS Create the global stiffness matrix K for a SCALAR problem in SERIAL computing.
%   StiffMa_sss            - STIFFMASS Create the global stiffness matrix tril(K) for a SCALAR problem in SERIAL computing
%   eStiff_spsa            - HEX8SCALARSAP Compute all tril(ke) for a SCALAR problem in PARALLEL computing
%   eStiff_ss              - HEX8SCALARS Compute the element stiffnes matrix for a SCALAR problem in SERIAL computing.
%   eStiff_sss             - HEX8SCALARSS Compute the lower symmetric part of the element stiffness matrix
%   eStiff_sssa            - HEX8SCALARSAS Compute the lower symmetric part of all ke in SERIAL computing

%% Script Files used to run the functions
%   runScalarCPUvsGPU      - Script to run the whole assembly code on the CPU and GPU, and compare them
%   runScalarOnCPU         - run the whole assembly code on the CPU
%   runScalarOnGPU         - run the whole assembly code on the GPU
%   runHex8ScalarCPUvsGPU  - Script to run the HEX8 (ke) code on the CPU and GPU, and compare them
%   runHex8ScalarOnCPU     - Script to run the HEX8 (ke) code on the CPU
%   runHex8ScalarOnGPU     - Script to run the HEX8 (ke) code on the GPU
%   runIndexScalarCPUvsGPU - Script to run the INDEX code on the CPU and GPU, and compare them
%   runIndexScalarOnCPU    - Script to run the INDEX code alone on the CPU
%   runIndexScalarOnGPU    - Script to run the INDEX code alone on the GPU
