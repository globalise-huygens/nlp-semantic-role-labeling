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
modelnames=('globalise/GloBERTise') #'FacebookAI/xlm-roberta-base'  #'google-bert/bert-base-multilingual-cased'
model_types=('RoBERTa') #XLM-R #BERT
data_directory=('path_to_data_files') 

DEST="path_to_save_results_on_Snellius " #where to save results on Snellius
mkdir -p "$DEST"

for learningrate in "${learningrates[@]}"
do
    for epoch in "${epochs[@]}"
    do
        for batch_size in "${batch_sizes[@]}"
        do
            for modelname in "${modelnames[@]}"
            do
                for model_type in "${model_types[@]}"
                do
                    for directory in "${data_directory[@]}"
                    do
                python fine_tune_SRL.py --learning_rate=$learningrate \
                                        --epoch=$epoch \
                                        --batch_size=$batch_size \
                                        --model_checkpoint=$modelname \
                                        --model_type=$model_type \
                                        --directory=$directory

                cp $TMPDIR/outputs/*.json "$DEST" 2>/dev/null
                cp $TMPDIR/outputs/*.csv "$DEST" 2>/dev/null
                cp $TMPDIR/outputs/*.txt "$DEST" 2>/dev/null
                cp $TMPDIR/outputs/*.png "$DEST" 2>/dev/null 

                # Checkpoint (already limited to one)
                cp -r $TMPDIR/outputs/checkpoint-* "$DEST" 2>/dev/null

                done
            done
        done
    done
done 


