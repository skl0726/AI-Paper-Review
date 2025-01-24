#!/bin/sh


#SBATCH -J test # job name
#SBATCH -o test.out # name of stdout output file (%j expands to %jobId)
#SBATCH -e test.err # define file name of standard error
#SBATCH -t 00:30:00 # run time (hh:mm:ss) 

#SBATCH -p titanxp # queue or partiton name
#SBATCH -N 1 # total number of needed computing nodes
#SBATCH -n 1 # number of nodes (total number of needed processes)
#SBATCH --nodelist=n1
#SBATCH  --gres=gpu:2 # gpus per node


cd $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"


module load cuda/11.2
module load cuDNN/cuda/11.2/8.1.0.77


echo "Start"
echo "condaPATH"

echo "source(path): $HOME/anaconda3/etc/profile.d/conda.sh"
source $HOME/anaconda3/etc/profile.d/conda.sh # path


echo "conda activate gpu"
conda activate gpu # conda env to use

srun python3 $HOME/temp/Deformable-DETR/train_practice.py
#python3 src/CUB/inference.py -model_dirs CUB_noisy_processed/seed0.7/sam_modelXtoC/best_model_1.pth CUB_noisy_processed/seed0.7/sam_modelXtoC/best_model_2.pth CUB_noisy_processed/seed0.7/sam_modelXtoC/best_model_3.pth -model_dirs2 CUB_noisy_processed/seed0.7/independent_sam/best_model_1.pth CUB_noisy_processed/seed0.7/independent_sam/best_model_2.pth CUB_noisy_processed/seed0.7/independent_sam/best_model_3.pth -eval_data test -use_attr -n_attributes 112 -data_dir CUB_noisy_processed/seed0.7/processed_data -bottleneck -use_sigmoid # python command
date

echo "conda deactivate"
conda deactivate # deactivate


squeue --job $SLURM_JOBID


echo "##### END #####"