#!/bin/bash
 
#########################
## SLURM JOB COMMANDS ###
#########################
#SBATCH --partition=maxgpu              # or allgpu
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --job-name test_job           # give job unique name
#SBATCH --output ./training-%j.out      # terminal output
#SBATCH --error ./training-%j.err       # error output
#SBATCH --mail-type END
#SBATCH --mail-user henning.rose@desy.de  # change to your mail address
#SBATCH --constraint=GPU                # or GPUx1, GPUx2, GPUx4 fore 1, 2 or 4 GPUs
#SBATCH --chdir=/beegfs/desy/user/rosehenn/first_training      # change to your workspace folder e.g. /beegfs/desy/user/...
 
##SBATCH --nodelist=max-cmsg004         # you can select specific nodes, if necessary
##SBATCH --constraint="P100|V100"       # ask for specific types of GPUs
##SBATCH --exclude=max-cmsg007          # you can exclude specific nodes if necessary
 
 
 
 
#####################
### BASH COMMANDS ###
#####################
 
## examples:
 
# source .bashrc for init of anaconda
source ~/.bashrc
 
# load modules (likely not necessary)
#source /etc/profile.d/modules.sh
#module load maxwell
#module load cuda
#module load anaconda3
 
# activate your conda environment the job should use
cd /home/rosehenn/first_training

singularity exec --nv -B /home -B /beegfs /beegfs/desy/user/birkjosc/singularity_images/pytorch-image-v0.0.4.img \
    bash -c "source /opt/conda/bin/activate && python test_job.py"
