#!/bin/bash
#SBATCH --nodes=1:ppn=4:gpus=1:titan
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=MNKnnets
#SBATCH --mail-type=END
#SBATCH --mail-user=gvg218@nyu.edu
#SBATCH --output=MNKnnet_%j.out

module purge
module load python/intel/3.5.1
source $HOME/nnet/bin/activate

RUNDIR=$SCRATCH/gvg218/nnetrundir/run-${SLURM_JOB_ID/.*}
mkdir RUNDIR

cd $RUNDIR

python $SCRATCH/gvg218/nnets-for-mnk/factorial_experiment.py
