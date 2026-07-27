#!/bin/bash                                                                                                     
#SBATCH   --job-name=Index_test                                                                                         
#SBATCH   --partition=GPU                                                                                     
#SBATCH   --nodes=1                                                                                               
#SBATCH   --ntasks-per-node=1                                                                                     
#SBATCH   --cpus-per-task=1                                                                                     
#SBATCH   --time=01-01:01                                                                                         

ssh nodo-g1
cd StiffMa 

module purge
module load Matlab/R2025b
matlab -nodesktop -nosplash
addpath(genpath(pwd))


