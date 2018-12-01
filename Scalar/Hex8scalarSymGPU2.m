%  * ====================================================================*/
% ** This function was developed by:
%  *          Francisco Javier Ramirez-Gil
%  *          Universidad Nacional de Colombia - Medellin
%  *          Department of Mechanical Engineering
%  *
%  ** Please cite this code as:
%  *
%  ** Date & version
%  *      30/11/2018.
%  *      V 1.2
%  *
%  * ====================================================================*/

function KE = Hex8scalarSymGPU2(elements,nodes,c)
% NNZ values of symmetric part of the element stiffness matrix ke (SCALAR)

nel = size(elements,1);                              % Number of elements
nnod = size(nodes,1);                                % Number of nodes
L = dNdrst;                                          % Shape functions derivatives in natural coord.
KE = zeros(36*nel,1,'double','gpuArray');            % Stores ke entries (initialized directly on GPU)
ker = parallel.gpu.CUDAKernel('Hex8scalarSymGPU2.ptx',...
    'const unsigned int *, const double *, const unsigned int, const unsigned int, const double, double *',...
    'Hex8scalarId'); % Kernel creation
ker.ThreadBlockSize = [256, 1, 1];                       % Threads per block
ker.GridSize = [ceil(nel/ker.ThreadBlockSize(1)), 1, 1]; % Blocks per grid
setConstantMemory(ker, 'L', L);
KE = feval(ker, elements', nodes', nel, nnod, c, KE);      % GPU code execution
