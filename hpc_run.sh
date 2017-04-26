#!/bin/bash
#PBS --nodes=1:ppn=4:gpus=1:titan
#PBS --ntasks=1
#PBS --time=48:00:00
#PBS --mem=8GB
#PBS --job-name=MNKnnets
#PBS --mail-type=END
#PBS --mail-user=gvg218@nyu.edu
#PBS --output=MNKnnet_%j.out

module purge
module load python/intel/3.5.1
source $HOME/nnet/bin/activate

RUNDIR=$SCRATCH/gvg218/nnetrundir/run-${SLURM_JOB_ID/.*}
mkdir RUNDIR

cd $RUNDIR

python $SCRATCH/gvg218/nnets-for-mnk/factorial_experiment.py
