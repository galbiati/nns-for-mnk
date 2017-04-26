#!/bin/bash
#PBS -l nodes=1:ppn=4:gpus=1:titan
#PBS -l walltime=48:00:00
#PBS -l mem=8GB
#PBS -N MNKnnets
#PBS -M gvg218@nyu.edu

module purge
module load python/intel/3.5.1
source $HOME/nnet/bin/activate

RUNDIR=$SCRATCH/gvg218/nnetrundir/run-${SLURM_JOB_ID/.*}
mkdir RUNDIR

cd $RUNDIR

python $SCRATCH/gvg218/nnets-for-mnk/factorial_experiment.py
