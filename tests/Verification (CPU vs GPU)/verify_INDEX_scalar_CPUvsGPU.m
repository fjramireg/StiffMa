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
    1 5 9 8 10 14 18 17;        % Element 1
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

%% INDEX iK, jK for the Stiffness matrix generation on the CPU

% MATLAB Computation on serial CPU (host)
[iKh, jKh] = Index_sssa(Mesh.elements, sets);               % indices for tril(K). Size: 36-by-nel, 1. (REFERENCE)

% MATLAB Computation on parallel CPU (host. Vectorized version)
[iKh_vec, jKh_vec] = Index_ssat(Mesh.elements, sets);       % indices for tril(K). Size: nel-by-36

% Comparison CPU vs CPU optimized
if ( sum(iKh==iKh_vec) ~= length(iKh) ) || ( sum(jKh==jKh_vec) ~= numel(jKh) )
    error('Mismatch in indices between CPU and vectorized CPU computations.');
else
    disp("Indices between CPU and vectorized CPU are correctly computed.")
    figCPU = figure('color','none');
    axCPU = axes('Parent',figCPU,'Color','none');
    hold on;
    spy(sparse(iKh,jKh,true),'or');         % K_tril from MATLAB Computation on serial CPU
    spy(sparse(iKh_vec,jKh_vec,true),'.b'); % K_tril from MATLAB Computation on parallel CPU (host. Vectorized version)
    title("CPU vs CPU optimized");
    legend('Serial CPU','Vectorized CPU');
    hold off;
end

%% INDEX iK, jK for the Stiffness matrix generation on the GPU
v = ver;
if ( any(strcmp({v.Name}, 'Parallel Computing Toolbox')) && (gpuDeviceCount > 0) )

    % GPU Settings
    d = gpuDevice;
    sets.tbs      = d.MaxThreadsPerBlock;   % Max. Thread Block Size
    sets.numSMs   = d.MultiprocessorCount;  % Number of multiprocessors on the device
    sets.WarpSize = d.SIMDWidth;            % The warp size in threads
    elementsGPU = gpuArray(Mesh.elements);  % Transfer from host-to-device memories

    % MATLAB Computation on parallel GPU (device. Old version)
    [iKd, jKd] = Index_spsa(elementsGPU, sets);  % indices for tril(K). Size: 36-by-nel, 1

    % MATLAB Computation on parallel GPU (device. Optimized version)
    [iKd_opt, jKd_opt] = Index_spsa_opt(elementsGPU, sets);     % indices for tril(K). Size: nel-by-36, 1
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

    % Comparison: CPU vs GPU
    % iKd = gather(iKd);
    % jKd = gather(jKd);
    if ( sum(iKd==iKh) ~= numel(iKh) || sum(jKd==jKh) ~= numel(jKh) )
        error('Mismatch in indices between CPU and GPU computations.');
    else
        disp("Indices between CPU and GPU (old version) are correctly computed.")
        figGPU = figure('color','none');
        axGPU = axes('Parent',figGPU,'Color','none');
        hold on;
        spy(sparse(iKh,jKh,true),'or');         % K_tril from MATLAB Computation on serial CPU
        spy(sparse(iKh_vec,jKh_vec,true),'.b'); % K_tril from MATLAB Computation on parallel GPU (device. Old version)
        title("CPU vs GPU");
        legend('Serial CPU','Parallel GPU (Old version)');
        hold off;
    end

    % Comparison: CPU vs GPU optimized
    if ( sum(iKd_opt2==iKh) ~= numel(iKh) || sum(jKd_opt2==jKh) ~= numel(jKh) )
        error('Mismatch in indices between CPU and reordered-GPU computations.');
    else
        disp("Indices between CPU and GPU (Optimized version) are correctly computed.")
        figGPU2 = figure('color','none');
        axGPU2 = axes('Parent',figGPU2,'Color','none');
        hold on;
        spy(sparse(iKh,jKh,true),'or');         % K_tril from MATLAB Computation on serial CPU
        spy(sparse(iKd_opt2,jKd_opt2,1),'.b');  % K_tril from MATLAB Computation on parallel GPU (device. Optimized version)
        title("CPU vs GPU");
        legend('Serial CPU','Parallel GPU (Optimized version)');
        hold off;
    end

    % Comparison: CPU vs unordered-GPU optimized version
    figGPU3 = figure('color','none');
    axGPU3 = axes('Parent',figGPU3,'Color','none');
    hold on;
    spy(sparse(iKh,jKh,true),'or');       % K_tril from MATLAB Computation on serial CPU
    spy(sparse(iKd_opt,jKd_opt,1),'.b');  % K_tril from MATLAB Computation on parallel GPU (device. Optimized version)
    title("CPU vs GPU");
    legend('Serial CPU','Parallel GPU (Optimized unordered version)');
    hold off;

end
