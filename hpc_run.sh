#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=6:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=MNKnnetTest
#SBATCH --mail-type=END
#SBATCH --mail-user=gvg218@nyu.edu
#SBATCH --output=mnk_%j.out

module purge
module load python/intel/3.5.1
# source $HOME/nnet/bin/activate #fix this!

cd /scratch/gvg218/nns-for-mnk
