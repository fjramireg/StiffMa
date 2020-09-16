% Inputs
nel = 170;
sets.sf = 1;
sets.dTE = 'uint32';
sets.dTN = 'double';
MP.E = 200e9;
MP.nu = 0.3;
ct = 384.1;

% For CPU assembly
[Iar, tnel, tdof] = Index_sosa(nel, nel, nel);
Ke = eStiff_sosa(ct, tnel);

%% Index CPU
[Indx, nels, tdofs] = Index_sosa(nel, nel, nel);

%% Local ke CPU
Keall = eStiff_sosa(ct, tnel);

%% Assembly CPU
K = AssemblyStiffMa_CPUo(Iar(:,1), Iar(:,2), Ke, tdof);
