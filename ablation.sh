#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -t 4-00:00:00
#SBATCH --mem=128GB
#SBATCH -J ablation_study
#SBATCH -o ablation_%j.out
#SBATCH -e ablation_%j.err

# Activate environment
source /share/apps/NYUAD/miniconda/3-4.11.0/bin/activate
conda activate sciurus

# Change to working directory
cd /scratch/an3854/aime_path_patching/new

# Parameters
MODEL="qwen-base"  # qwen-chat, qwen-instruct, deepseek-distilled, qwen-base, llama-base, llama-instruct, llama-distilled, open-reasoner, llama-rl
DATASET="trivia"      # math, trivia
NUM_PROBLEMS=1000
BATCH_SIZE=1000

echo "Starting ablation study..."
echo "Model: $MODEL, Dataset: $DATASET, Problems: $NUM_PROBLEMS, Batch Size: $BATCH_SIZE"

# Layer-by-layer ablation
echo "Running layer-by-layer ablation..."
python main.py \
    --model $MODEL \
    --dataset $DATASET \
    --num_problems $NUM_PROBLEMS \
    --batch_size $BATCH_SIZE \
    --ablation_type layer

echo "Ablation study completed."

# Uncomment to run other ablation types:
# Head ablation
# python main.py --model $MODEL --dataset $DATASET --ablation_type head --layer_idx 8 --head_idx 12

# Layer range ablation  
# python main.py --model $MODEL --dataset $DATASET --ablation_type layer_range --start_layer 5 --end_layer 10

# Layer permutation
# python main.py --model $MODEL --dataset $DATASET --ablation_type permute --start_layer 3 --end_layer 8