% Inputs
nel = 110;
sets.sf = 1;
sets.dTE = 'uint32';
sets.dTN = 'double';
MP.E = 200e9;
MP.nu = 0.3;
ct = 384.1;

% Mesh generation
[elements, ~] = CreateMesh2(nel, nel, nel, sets);

% For CPU assembly
Ke = eStiff_sosa2(ct);
opts.symmetric = 1;
opts.n_node_dof = 1;

%% Local ke CPU
Ke = eStiff_sosa2(ct);

%% Assembly CPU
K = AssemblyStiffMa_CPUo2(elements', Ke, opts);
