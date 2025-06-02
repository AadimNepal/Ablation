#!/bin/bash
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH -t 12-00:00:00
#SBATCH --mem=128GB
#SBATCH -J mmlu_comprehensive
#SBATCH -o mmlu_comprehensive_%j.out
#SBATCH -e mmlu_comprehensive_%j.err

# Activate environment
source /share/apps/NYUAD/miniconda/3-4.11.0/bin/activate
conda activate sciurus

# Change to working directory
cd /scratch/an3854/aime_path_patching/new

# ============================================================================
# COMPREHENSIVE MMLU ABLATION FOR ALL MODELS
# Tests all models on both factual and reasoning questions
# ============================================================================

# Define all models to test
MODELS=("qwen-chat" "qwen-instruct" "deepseek-distilled" "qwen-base" "llama-base" "llama-instruct" "llama-distilled" "open-reasoner" "llama-rl")

# Define question types to test
QUESTION_TYPES=("factual" "reasoning")

# Parameters for comprehensive testing
NUM_PROBLEMS=1000
BATCH_SIZE=1000
DATASET="mmlu"

echo "============================================================================"
echo "STARTING COMPREHENSIVE MMLU ABLATION STUDY"
echo "Models: ${MODELS[@]}"
echo "Question Types: ${QUESTION_TYPES[@]}"
echo "Problems per experiment: $NUM_PROBLEMS"
echo "Batch size: $BATCH_SIZE"
echo "Total experiments: $((${#MODELS[@]} * ${#QUESTION_TYPES[@]}))"
echo "============================================================================"

# Function to run experiment with error handling
run_experiment() {
    local model="$1"
    local question_type="$2"
    local experiment_name="${model}_${question_type}"
    
    echo ""
    echo "=========================================="
    echo "EXPERIMENT: $experiment_name"
    echo "Model: $model"
    echo "Question Type: $question_type"
    echo "Started at: $(date)"
    echo "=========================================="
    
    # Run the experiment
    python main.py \
        --model "$model" \
        --dataset "$DATASET" \
        --num_problems "$NUM_PROBLEMS" \
        --batch_size "$BATCH_SIZE" \
        --ablation_type layer \
        --mmlu_question_type "$question_type"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ SUCCESS: $experiment_name completed successfully"
    else
        echo "❌ FAILED: $experiment_name failed with exit code $exit_code"
    fi
    
    echo "Finished at: $(date)"
    echo "=========================================="
    
    return $exit_code
}

# Track experiment results
successful_experiments=0
failed_experiments=0
experiment_log="experiment_results_$(date +%Y%m%d_%H%M%S).log"

echo "Experiment log will be saved to: $experiment_log"
echo "Experiment,Model,QuestionType,Status,Timestamp" > "$experiment_log"

# Main experiment loop
experiment_count=0
total_experiments=$((${#MODELS[@]} * ${#QUESTION_TYPES[@]}))

for model in "${MODELS[@]}"; do
    for question_type in "${QUESTION_TYPES[@]}"; do
        experiment_count=$((experiment_count + 1))
        
        echo ""
        echo "🚀 STARTING EXPERIMENT $experiment_count / $total_experiments"
        
        # Run the experiment
        if run_experiment "$model" "$question_type"; then
            successful_experiments=$((successful_experiments + 1))
            echo "$experiment_count,$model,$question_type,SUCCESS,$(date)" >> "$experiment_log"
        else
            failed_experiments=$((failed_experiments + 1))
            echo "$experiment_count,$model,$question_type,FAILED,$(date)" >> "$experiment_log"
            
            # Optional: Continue with next experiment even if one fails
            echo "⚠️  Continuing with next experiment despite failure..."
        fi
        
        # Optional: Add delay between experiments to let GPU cool down
        echo "Pausing for 30 seconds between experiments..."
        sleep 30
    done
done

# Final summary
echo ""
echo "============================================================================"
echo "COMPREHENSIVE MMLU ABLATION STUDY COMPLETED"
echo "============================================================================"
echo "Total experiments run: $experiment_count"
echo "Successful experiments: $successful_experiments"
echo "Failed experiments: $failed_experiments"
echo "Success rate: $(( (successful_experiments * 100) / experiment_count ))%"
echo "Results log: $experiment_log"
echo "Completed at: $(date)"
echo "============================================================================"

# Display experiment log
echo ""
echo "EXPERIMENT SUMMARY:"
echo "-------------------"
cat "$experiment_log"

# Check results directory
echo ""
echo "RESULTS DIRECTORIES CREATED:"
echo "----------------------------"
find results/ -type d -name "*mmlu*" 2>/dev/null | head -20

echo ""
echo "🎉 All experiments completed!"
echo "Check the results/ directory for detailed outputs from each experiment."
echo "Check $experiment_log for a summary of which experiments succeeded/failed."