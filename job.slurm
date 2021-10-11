#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=ie-idi
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500
#SBATCH --job-name="test"
#SBATCH --output=test.out
#SBATCH --mail-user=bjornar.osttveit@ntnu.no
#SBATCH --mail-type=NONE

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

module purge
module load Anaconda3/2020.07
python ./test.py
