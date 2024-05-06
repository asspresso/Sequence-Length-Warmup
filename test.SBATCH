#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --time=2:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:rtx8000
#SBATCH --job-name=torch

module purge

singularity exec --nv \
	    --overlay /scratch/rw2608/pytorch-example/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; cd /vast/rw2608/warmup; python Sequence_Length_Warmup_6.py"
