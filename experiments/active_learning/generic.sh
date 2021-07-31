#!/bin/bash

# This is a generic running script. It can run in two configurations:
# Single job mode: pass the python arguments to this script
# Batch job mode: pass a file with first the job tag and second the commands per line

#! Name of the job:
#SBATCH -J testjob

#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#! mlmi8 acocunt: SBATCH -A MLMI-CLM88-SL2-CPU
#SBATCH -A T2-CS133-GPU

#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#SBATCH --ntasks=1

#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1).
#! Note that the job submission script will enforce no more than 3 cpus per GPU.
#SBATCH --gres=gpu:1
#! SBATCH --cpus-per-task=4

#! Send mail since queue times are long
#SBATCH --mail-type=FAIL,BEGIN,END

#! Which GPU/CPU nodes to run on (pascal for GPU; skylake,cclake for CPU)
#SBATCH -p pascal

#SBATCH --time=05:00:00

#! ############################################################
set -e # fail fully on first line failure

# Customize this line to point to conda installation
path_to_conda="./miniconda3"

echo "Running on $(hostname)"

if [ -z "$SLURM_ARRAY_TASK_ID" ]
then
    # Not in Slurm Job Array - running in single mode

    JOB_ID=$SLURM_JOB_ID

    # Just read in what was passed over cmdline
    JOB_CMD="${@}"
else
    # In array

    JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

    # Get the line corresponding to the task id
    JOB_CMD=$(head -n ${SLURM_ARRAY_TASK_ID} "$1" | tail -1)
fi

# Find what was passed to --output_folder
regexp="--output_folder\s+(\S+)"
if [[ $JOB_CMD =~ $regexp ]]
then
    JOB_OUTPUT=${BASH_REMATCH[1]}
else
    echo "Error: did not find a --output_folder argument"
    exit 1
fi

# Check if results exists, if so remove slurm log and skip
if [ -f  "$JOB_OUTPUT/results.json" ]
then
    echo "Results already done - exiting"
    rm "slurm-${JOB_ID}.out"
    exit 0
fi

# Check if the output folder exists at all. We could remove the folder in that case.
if [ -d  "$JOB_OUTPUT" ]
then
    echo "Folder exists, but was unfinished or is ongoing (no results.json)."
    echo "Starting job as usual"
    # It might be worth removing the folder at this point:
    # echo "Removing current output before continuing"
    # rm -r "$JOB_OUTPUT"
    # Since this is a destructive action it is not on by default
fi

# Use this line if you need to create the environment first on a machine
# ./run_locked.sh ${path_to_conda}/bin/conda-env update -f environment.yml

# Activate the environment
#source ${path_to_conda}/bin/activate example-environment # conda version
source ../../env/bin/activate # venv version

# Train the model
echo "Job: ${JOB_CMD}"
srun python3 $JOB_CMD

# Move the log file to the job folder
mv "slurm-${JOB_ID}.out" "${JOB_OUTPUT}/"
