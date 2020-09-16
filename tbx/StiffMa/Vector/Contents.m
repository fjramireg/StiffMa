% VECTOR folder contains all the necessary code to assembly global sparse
% stiffness matrices from finite element analysis of vector problems like
% structural phenomena.
% Version 1.4 31-Jan-2020
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
%   DMatrix     - Compute the isotropic material matrix for the VECTOR problem.

%% --- INDEX COMPUTATION ---
%   Index_va    - Computes the row/column indices of K for a VECTOR (s) problem on the
%   Index_vpsa  - Compute the row/column indices of tril(K) in a vector (v) problem
%   Index_vsa   - Computes the row/column indices of K for a VECTOR (s) problem using
%   Index_vssa  - Compute the row/column indices of tril(K) in a vector (v) problem
%   eStiff_vpsa - Compute the element stiffness matrices for a VECTOR (v) problem
%   eStiff_vsa  - Computes the element stiffness matrices for a VECTOR (v) problem
%   eStiff_vssa - ESTIFFA_VSS Compute ALL (a) the element stiffness matrices for a VECTOR (v)

%% --- ELEMENT STIFFNESS COMPUTATION ---
%   eStiff_vs   - Compute the element stiffness matrix for a VECTOR (s) problem
%   eStiff_vss  - Compute the element stiffness matrix for a VECTOR (v) problem in

%% --- GLOBAL STIFFNESS COMPUTATION ---
%   StiffMa_vps - Create the global stiffness matrix for a VECTOR (v) problem
%   StiffMa_vs  - Create the global stiffness matrix K for a VECTOR (v) problem
%   StiffMa_vss - Create the global stiffness matrix for a VECTOR (v) problem

