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
%
% --- INDEX COMPUTATION ---
%   Index_sss              - Computes the row/column indices of tril(K) for a SCALAR (s)
%   Index_sps              - Computes the row/column indices of tril(K) for a SCALAR (s)
%
% --- ELEMENT STIFFNESS COMPUTATION ---
%   eStiff_ss              - Computes the element stiffnes matrix for a SCALAR (s) problem
%   eStiff_sss             - Computes the element stiffness matrix for a SCALAR (s) problem
%   eStiff_sssa            - Computes the element stiffness matrices for a SCALAR (s)
%   eStiff_spsa            - Computes the element stiffness matrices for a SCALAR (s)
%
% --- GLOBAL STIFFNESS COMPUTATION ---
%   StiffMa_ss             - Create the global stiffness matrix K for a SCALAR (s) problem
%   StiffMa_sss            - Create the global stiffness matrix tril(K) for a SCALAR (s)
%   StiffMa_sps            - Create the global stiffness matrix tril(K) for a SCALAR (s)

%% Script Files used to run the functions
%
% --- INDEX COMPUTATION ---
%   runIndexScalarOnCPU    - Runs the INDEX scalar code on the CPU
%   runIndexScalarOnGPU    - Runs the INDEX scalar code on the GPU
%   runIndexScalarCPUvsGPU - Runs the INDEX scalar code. CPU vs GPU
%
% --- ELEMENT STIFFNESS COMPUTATION ---
%   runHex8ScalarOnCPU     - Runs the HEX8 scalar code on the CPU
%   runHex8ScalarOnGPU     - Runs the HEX8 scalar code on the GPU
%   runHex8ScalarCPUvsGPU  - Runs the HEX8 scalar code. CPU vs GPU
%
% --- GLOBAL STIFFNESS COMPUTATION ---
%   runScalarOnCPU         - Runs the whole assembly scalar code on the CPU
%   runScalarOnGPU         - Runs the whole assembly scalar code on the GPU
%   runScalarCPUvsGPU      - Runs the whole assembly scalar. CPU vs GPU
