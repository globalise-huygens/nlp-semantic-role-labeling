#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=8:00:00
#SBATCH --output=finetune.out

module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

#module load 2023
#module load Python/3.12.3-GCCcore-13.3.0
#module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1


learningrates=(5e-5)
epochs=(30)
batch_sizes=(16)
modelnames=('google-bert/bert-base-multilingual-cased') #'emanjavacas/GysBERT-v2' #'FacebookAI/xlm-roberta-base' 

for learningrate in "${learningrates[@]}"
do
    for epoch in "${epochs[@]}"
    do
    for batch_size in "${batch_sizes[@]}"
        do
            for modelname in "${modelnames[@]}"
            do
            
            python New_Multitask.py --learning_rate=$learningrate \
                                    --epoch=$epoch \
                                    --batch_size=$batch_size\
                                    --model_checkpoint=$modelname 
            done
        done
    done
done 
