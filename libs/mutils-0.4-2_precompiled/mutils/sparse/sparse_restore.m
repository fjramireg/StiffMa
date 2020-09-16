function [A, Ai, Aj, Ax] = sparse_restore(Ac)
%SPARSE_RESTORE converts a sparse matrix created by SPARSE_CONVERT back
%to MATLABs native sparse matrix.
%
%Undocumented

% Copyright 2012, Marcin Krotkiewski, University of Oslo

if isempty(Ac.thread_Aj)
  error('Can not resotre matrix. Ac.thread_Aj field is empty.')
end

Ai = [];
Aj = [];

nthreads = length(Ac.nz_cpu_dist);
if ~Ac.localized
  for thr=1:nthreads
    Ai = [Ai Ac.thread_Ai{thr}+1];
    Aj = [Aj Ac.thread_Aj{thr}+1];
  end
else
  for thr=1:nthreads
    Ai = [Ai Ac.thread_Ai{thr}+1+(Ac.row_cpu_dist(thr)-Ac.local_offset(thr))];
    Aj = [Aj Ac.thread_Aj{thr}+1+(Ac.row_cpu_dist(thr))];
  end
end

if Ac.block_size==1
  Ax = [Ac.thread_Ax{:}];
else
  Ax = ones(size(Ai));
end
A = sparse(double(Ai), double(Aj), Ax);

end
