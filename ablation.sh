#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -t 4-00:00:00
#SBATCH --mem=128GB
#SBATCH -J ablation_study
#SBATCH -o ablation_%j.out
#SBATCH -e ablation_%j.err

# Load required modules
module load gcc
export CC=gcc

# Activate environment
source /share/apps/NYUAD/miniconda/3-4.11.0/bin/activate
conda activate /scratch/ss13750/sciurus

# Change to working directory
cd /scratch/ss13750/aadim/Ablation

# Parameters
MODEL="open-reasoner" # qwen-chat, qwen-instruct, deepseek-distilled, qwen-base, llama-base, llama-instruct, llama-distilled, open-reasoner, llama-rl
DATASET="mmlu" # math, trivia, mmlu
NUM_PROBLEMS=1000
BATCH_SIZE=1000 # Reduced for better memory management
# MMLU-specific parameters (only used when DATASET="mmlu")
MMLU_QUESTION_TYPE="reasoning" # "", "factual", or "reasoning"
MMLU_SUBJECTS="" # "", or specific subjects like "college_mathematics high_school_physics"

echo "Starting ablation study..."
echo "Model: $MODEL, Dataset: $DATASET, Problems: $NUM_PROBLEMS, Batch Size: $BATCH_SIZE"

# Build the command based on dataset
if [ "$DATASET" = "mmlu" ]; then
    echo "MMLU Configuration:"
    echo " Question Type Filter: ${MMLU_QUESTION_TYPE:-'all'}"
    echo " Subject Filter: ${MMLU_SUBJECTS:-'all subjects'}"
    
    # Build MMLU command
    MMLU_ARGS=""
    if [ ! -z "$MMLU_QUESTION_TYPE" ]; then
        MMLU_ARGS="$MMLU_ARGS --mmlu_question_type $MMLU_QUESTION_TYPE"
    fi
    if [ ! -z "$MMLU_SUBJECTS" ]; then
        MMLU_ARGS="$MMLU_ARGS --mmlu_subjects $MMLU_SUBJECTS"
    fi
    
    echo "Running layer-by-layer ablation on MMLU..."
    python main.py \
        --model $MODEL \
        --dataset $DATASET \
        --num_problems $NUM_PROBLEMS \
        --batch_size $BATCH_SIZE \
        --ablation_type layer \
        $MMLU_ARGS
else
    # Original command for math/trivia
    echo "Running layer-by-layer ablation on $DATASET..."
    python main.py \
        --model $MODEL \
        --dataset $DATASET \
        --num_problems $NUM_PROBLEMS \
        --batch_size $BATCH_SIZE \
        --ablation_type layer
fi

echo "Ablation study completed."

# Uncomment to run other ablation types:
# Head ablation
# python main.py --model $MODEL --dataset $DATASET --ablation_type head --layer_idx 8 --head_idx 12

# Layer range ablation
# python main.py --model $MODEL --dataset $DATASET --ablation_type layer_range --start_layer 5 --end_layer 10

# Layer permutation
# python main.py --model $MODEL --dataset $DATASET --ablation_type permute --start_layer 3 --end_layer 8