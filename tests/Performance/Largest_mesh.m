syms nel sz szInd szNNZ AM 

Mmesh = szInd*(8 * (nel^3)) + szNNZ*(3 * (nel+1)^3 );

Mtrip = (2*szInd + szNNZ)*sz*(nel^3);

Mtotal = Mmesh + 4.5*Mtrip;

Ndiv = Mtotal/AM;

% MTotc = Mmesh + Mtrip*(4*Ndiv + 0.5);
% MTotc = Mmesh + 0.5*Mtrip;   % for Ndiv -> inf
% 
syms MTotc
% eqn = Mmesh + Mtrip*(4*Ndiv + 0.5) - MTotc == 0;
eqn = Mmesh + 0.5*Mtrip - MTotc == 0;   % for Ndiv -> inf
Sol = isolate(eqn, nel)
% syms f(nel, sz, szInd, szNNZ, AM) 
% f = MTotc
% fe = subs(f, [szInd, szNNZ, sz, AM, nel], [4, 8, 36, 4e9, 160])

% assume(nel,{'positive','integer'});
assume(szInd,{'positive','integer'});
assume(szNNZ,{'positive','integer'});
assume(sz,{'positive','integer'})

% eqn1 = MTotc == Mmesh + Mtrip*(4*Ndiv + 0.5)
% eqn2 = isolate(eqn1, nel)

S = solve(MTotc,nel)
% S = solve(MTotc,nel,'ReturnConditions',true)
% S = solve(MTotc,nel,'ReturnConditions',true, 'MaxDegree', 3)
% nel_s = solve(MTotc,nel,'Real',true) 
% [nel_s,parameters,conditions] = solve(MTotc,nel,'ReturnConditions',true) 


% S1 = subs(S, [szInd, szNNZ, sz, AM], [4, 8, 36, 4e9])
% subs(S, sz,36,)
% subs(S, szNNZ,8)
