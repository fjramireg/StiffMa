nel = 96;


%% Input memory

% Connectivity
rows_e = nel^3;
cols_e = 8;
sizeof_uint32 = 4;   % Bytes
mem_elements = rows_e * cols_e * sizeof_uint32; % Bytes

% Nodal coordinates
rows_n = (nel+1)^3;
cols_n = 3;
sizeof_double = 8;   % Bytes
mem_nodes = rows_n * cols_n * sizeof_double; % Bytes

% Total input memory
input_mem = [mem_elements,  mem_nodes]/1e6;     % MBytes


%% Intermediate memory

% Scalar
rows_iK = 36;
cols_iK = nel^3;
mem_iK = rows_iK * cols_iK * sizeof_uint32;
mem_jK = mem_iK;
mem_Ke = rows_iK * cols_iK * sizeof_double;

% Vector
rowv_iK = 300;
colv_iK = nel^3;
memv_iK = rowv_iK * colv_iK * sizeof_uint32;
memv_jK = memv_iK;
memv_Ke = rowv_iK * colv_iK * sizeof_double;

% Total intermediate memory
inter_mem_sca = [mem_iK, mem_jK, mem_Ke]/1e6;       % MBytes
inter_mem_vec = [memv_iK, memv_jK,  memv_Ke]/1e6;   % MBytes


%% Output memory

% Mesh
sets.dTE = 'uint32';
sets.dTN = 'double';
sets.nel = nel^3;
[Mesh.elements, Mesh.nodes] = CreateMesh2(nel, nel, nel, sets.dTE, sets.dTN);

% Scalar
sets.sz = 36;
sets.edof = 8;
[iKs, jKs] = Index_sssa(Mesh.elements, sets);
% Kes = rand(rows_iK*cols_iK, 1);
% Ks = AssemblyStiffMa(iKs, jKs, rand(rows_iK*cols_iK, 1), sets.dTE, sets.dTN);
Ks = accumarray([iKs,jKs], rand(rows_iK*cols_iK, 1), [], [], [], 1);
dat = whos('Ks');
mem_outpu_sca = dat.bytes/1e6;                      % MBytes

% Vector
sets.sz = 300;
sets.edof = 24;
[iKv, jKv] = Index_vssa(Mesh.elements, sets);
Kv = accumarray([iKv,jKv], rand(rowv_iK*colv_iK, 1), [], [], [], 1);
datv = whos('Kv');
mem_outpu_vec = datv.bytes/1e6;                     % MBytes


%% Total memory

% Scalar
total_mem_sca =  (sum(input_mem) + sum(inter_mem_sca) + mem_outpu_sca);   % MBytes
mem_sca = [input_mem, inter_mem_sca, mem_outpu_sca, total_mem_sca];

% Vector
total_mem_vec =  (sum(input_mem) + sum(inter_mem_vec) + mem_outpu_vec);   % MBytes
mem_vec = [input_mem, inter_mem_vec, mem_outpu_vec, total_mem_vec];


