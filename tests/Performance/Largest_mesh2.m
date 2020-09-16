
%% Variables definition
syms nel sz szInd szNNZ AM 

%% Variable assumptions
% assume(nel,{'positive','integer'});
assume(szInd,{'positive','integer'});
assume(szNNZ,{'positive','integer'});
assume(sz,{'positive','integer'})
assume(AM,{'positive','Real'})

%% Equation 
Mmesh = szInd*(8 * (nel^3)) + szNNZ*(3 * (nel+1)^3 );       % Memory required for mesh storage
Mtrip = (2*szInd + szNNZ)*sz*(nel^3);                       % Memory required for triplet sparse format storage
eqn = Mmesh + 0.5*Mtrip - AM == 0;                          % Equation for Ndiv -> inf, MTotc -> AM

%% Solution
Sol = isolate(eqn, nel);                                    % Isolate 'nel' from equation
% Snumeric_sca = vpa(Sol)
% syms z; nel == root(sz*szNNZ*z^3 + 2*sz*szInd*z^3 + 6*szNNZ*z^3 + 16*szInd*z^3 + 18*szNNZ*z^2 + 18*szNNZ*z + 6*szNNZ - 2*AM, z, 1)
% 
% % R = solve(eqn,nel);
% % Rexplicit = solve(eqn,nel,'MaxDegree',3);
% % simplify(Rexplicit);
% % pretty(Rexplicit);

%% Substitution

% Help
% szInd = 4;  % Bytes for uint32 data type
% szNNZ = 8;  % Bytes for double data type
% AM = 4e9;   % Bytes of available memory on the GPU
% sz = 36;    % Number of entries for storing Ke in the SCALAR problem
% sz = 300;   % Number of entries for storing Ke in the VECTOR probles

% For scalar problem
S0 = subs(Sol, [szInd, szNNZ, sz], [4, 8, 36]); % For any AM
S1 = subs(Sol, [szInd, szNNZ, sz, AM], [4, 8, 36, 16e9]);
S1numeric_sca = vpa(S1)


% For scalar problem
S2 = subs(Sol, [szInd, szNNZ, sz, AM], [4, 8, 300, 16e9]);
S1numeric_vec = vpa(S2)
S3 = subs(Sol, [szInd, szNNZ, sz], [4, 8, 300]); % For any AM
