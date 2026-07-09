% This script is used to compare the results between CPU and GPU in MATLAB for the scalar problem.
% 
%   For more information, see the <a href="matlab:
%   web('https://github.com/fjramireg/StiffMa')">StiffMa</a> web site.
% 
%   Written by Francisco Javier Ramirez-Gil, fjramireg@gmail.com
%   Universidad Nacional de Colombia - Medellin
%   Institución Universitaria Pascual Bravo, Medellin-Colombia
%       Created: April 14, 2026. Version: 1.0


%% Mesh generation (connectivity array)
elements = [
    1 5 9 8 10 14 18 17;% Element 1
    5 2 6 9 14 11 15 18;        % Element 2
    9 6 3 7 18 15 12 16;        % Element 3
    8 9 7 4 17 18 16 13;        % Element 4
    10 14 18 17 19 23 27 26;    % Element 5
    14 11 15 18 23 20 24 27;    % Element 6
    18 15 12 16 27 24 21 25;    % Element 7
    17 18 16 13 26 27 25 22];   % Element 8

%% Settings
dTE = 'uint32';     % Data precision for "elements" ['uint32', 'uint64']
if strcmp(dTE,'uint32')
    Mesh.elements = uint32(elements);
else
    Mesh.elements = uint64(elements);
end
[nel, nxe] = size(Mesh.elements);
dxn = 1;            % For vector 3 (UX, UY, UZ). For scalar 1 (Temp)
sets.dTE = dTE;     % Data precision for computing
sets.nel = nel;     % Number of finite elements
sets.nxe = nxe;     % Number of nodes per element
sets.dxn = dxn;     % Number of DOFs per node 
sets.edof= dxn*nxe; % Number of DOFs per element 
sets.sz  = sets.edof * (sets.edof + 1) / 2; % Number of symmetry entries

%% GPU Settings
d = gpuDevice;
sets.tbs      = d.MaxThreadsPerBlock;   % Max. Thread Block Size
sets.numSMs   = d.MultiprocessorCount;  % Number of multiprocessors on the device
sets.WarpSize = d.SIMDWidth;            % The warp size in threads

%%  INDEX iK, jK for the Stiffness matrix generation

% MATLAB Computation on serial CPU (host)
[iKh, jKh] = Index_sssa(Mesh.elements, sets);               % indices for tril(K). Size: 36-by-nel, 1
% Organize in order to properly compare
[iKh2, jKh2] = find(sparse(iKh,jKh,true));

% MATLAB Computation on parallel CPU (host. Vectorized version)
[iKh_vec, jKh_vec] = Index_ssat(Mesh.elements, sets);       % indices for tril(K). Size: nel-by-36
% iKh_vec2 = reshape(iKh_vec.', [], 1);                     % i-index for tril(K). Size: 36-by-nel, 1
% jKh_vec2 = reshape(jKh_vec.', [], 1);                     % j-index for tril(K). Size: 36-by-nel, 1
[iKh_vec2, jKh_vec2] = find(sparse(iKh_vec,jKh_vec,true));

% % MATLAB Computation on parallel GPU (device. Old version)
[iKd, jKd] = Index_spsa(gpuArray(Mesh.elements'), sets);     % indices for tril(K). Size: 36-by-nel, 1

% MATLAB Computation on parallel GPU (device. Optimized version)
[iKd_opt, jKd_opt] = Index_spsa_opt(gpuArray(Mesh.elements), sets);     % indices for tril(K). Size: nel-by-36, 1
% Data should be reorganized to obtain the original MATLAB-style sz*e + t ordering, e.g. 
% Option 1:
% M = reshape(iKd, sets.nel, sets.sz);   % sets.nel-by-sets.sz
% T = M.';                                 % nonconjugate transpose
% iKd = T(:);                              % column vector
% M = reshape(jKd, sets.nel, sets.sz);   % sets.nel-by-sets.sz
% T = M.';                                 % nonconjugate transpose
% jKd = T(:);                              % column vector
% Option 2:
iKd_opt2 = reshape(reshape(iKd_opt, sets.nel, sets.sz).', [], 1); % i-index for tril(K). Size: nel-by-36, 1
jKd_opt2 = reshape(reshape(jKd_opt, sets.nel, sets.sz).', [], 1); % j-index for tril(K). Size: nel-by-36, 1

%% Comparison CPU vs CPU optimized

if ( sum(iKh2==iKh_vec2) ~= numel(iKh) ) || ( sum(iKd_opt2==iKh) ~= numel(iKh) ) || ( sum(iKh_vec2==iKh) ~= numel(iKh) )
    error('Mismatch in indices between CPU and GPU computations for iKd.');
elseif sum(jKd==jKh) ~= numel(jKh)
    error('Mismatch in indices between CPU and GPU computations for jKd.');
else
    disp("Indices between CPU and GPU are correctly computed.")
end

%% Comparison GPU vs GPU optimized
% 
% % iKd = gather(iKd);
% % jKd = gather(jKd);
% if ( sum(iKd==iKh) ~= numel(iKh) ) || ( sum(iKd_opt2==iKh) ~= numel(iKh) ) || ( sum(iKh_vec2==iKh) ~= numel(iKh) )
%     error('Mismatch in indices between CPU and GPU computations for iKd.');
% elseif sum(jKd==jKh) ~= numel(jKh)
%     error('Mismatch in indices between CPU and GPU computations for jKd.');
% else
%     disp("Indices between CPU and GPU are correctly computed.")
% end
% 
% 
% if sum(iKd==iKd_opt) ~= numel(iKd)
%     error('Mismatch in indices between GPU and GPU optimized computations for iKd_opt.');
% elseif sum(jKd==jKd_opt) ~= numel(jKd)
%     error('Mismatch in indices between GPU and GPU optimized computations for jKd_opt.');
% else
%     disp("Indices between GPU and GPU optimized are correctly computed.")
% end