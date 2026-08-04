#!/bin/bash
#SBATCH --job-name=Matlab_IndexTest       # Descriptive title of the work
#SBATCH --partition=GPU                   # Queue/Assigned Partition
#SBATCH --nodes=1                         # Number of nodes requested
#SBATCH --nodelist=NODO-G1                # Force SLURM to use ONLY the NODE-G3
#SBATCH --ntasks=1                        # A single main task (MATLAB)
#SBATCH --cpus-per-task=4                # Assign 64 physical CPUs to this task
## SBATCH --exclusive                       # Assigns the node exclusively (without sharing)
#SBATCH --gres=gpu:1                      # Requires 1 physical GPU (Tesla T4)
#SBATCH --time=24:00:00                   # Maximum execution time (adjust as needed)
#SBATCH --output=Index_LogResult_%j.out   # Standard output file (%j adds the Job ID)
#SBATCH --error=Index_%j.err              # Standard Error Log

# Notificación por Correo
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francisco.ramirez@pascualbravo.edu.co

# 1. Load the necessary environment and modules
# module avail
module purge
module load Matlab/R2025b

# 2. Print useful information to the output log (optional; useful for debugging)
echo "=== START OF WORK ==="
echo "Date and time: $(date)"
echo "Execution node: $SLURM_NODELIST"
echo "Job Directory: $SLURM_SUBMIT_DIR"
echo "=========================="

# 3. Run the code in MATLAB
matlab -batch 'runIndexTest2026'

# 4. Sent to the queue 
# using the command: sbatch batch_script_index.sl

# 5. Save the MATLAB output code
STATE=$?

# Send a notification to Discord based on the result
if [ $STATE -eq 0 ]; then
  MESSAGE="✅ **Project Successfully Completed**\n• **Job:** \`$SLURM_JOB_NAME\`\n• **Job ID:** \`$SLURM_JOB_ID\`\n• **Nodo:** \`$SLURM_NODELIST\`"
else
  MESSAGE="❌ **Job Failed** (Error code: $STATE)\n• **Job:** \`$SLURM_JOB_NAME\`\n• **Job ID:** \`$SLURM_JOB_ID\`\n• **Nodo:** \`$SLURM_NODELIST\`"
fi

curl -H "Content-Type: application/json" \
     -X POST \
     -d "{\"content\": \"$MESSAGE\"}" \
     https://discord.com/api/webhooks/1533852751396601876/RzuX6eb-a7Ac59ie_iDxsqi6MpziUTAXTbiY1HisQz0h5xMGAAtwU_bKQDNU4Cm4vhst  
