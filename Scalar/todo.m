matlab -nodisplay < myScript.m

delete(gcp('nocreate'));
pc = parcluster('local');
pc.JobStorageLocation = ‘~/scratch/.matlab/<jobID>';
pc.NumWorkers = <reqd number, if more than no of cores>;
parpool(pc, pc.NumWorkers);
feature('numcores')
numCores = feature('numcores');
p = parpool(numCores);
.
parfor . . .
.
.
end
.
delete(gcp('nocreate'));