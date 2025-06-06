import wandb
import argparse
import os
import json
from datetime import datetime

from vllm_model import (
    Qwen257BBase, Qwen257BInstruct, DeepSeekR1DistillQwen7B,
    Llama318BBase, Llama318BInstruct, DeepSeekR1DistilledLlama, 
    OpenReasonerBase, Llama31SimpleRLZoo
)
from gsm8 import GSM8KLoader, TriviaQALoader, MMLULoader, Math500Loader
from math_trainer import MathTrainer
from trivia_trainer import TriviaTrainer
from mmlu_trainer import MMLUTrainer
from math_500_trainer import Math500Trainer


def get_model(model_name):
    """Factory function to get model instance"""
    model_classes = {
        "qwen-instruct": Qwen257BInstruct,
        "deepseek-distilled": DeepSeekR1DistillQwen7B,
        "qwen-base": Qwen257BBase,
        "llama-base": Llama318BBase,
        "llama-instruct": Llama318BInstruct,
        "llama-distilled": DeepSeekR1DistilledLlama,
        "open-reasoner": OpenReasonerBase,
        "llama-rl": Llama31SimpleRLZoo
    }
    
    if model_name not in model_classes:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model_classes[model_name]()


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation study for LLMs")
    parser.add_argument("--model", type=str, default="qwen-chat",
                        choices=["qwen-instruct", "deepseek-distilled", "qwen-base", 
                                "llama-base", "llama-instruct", "llama-distilled", "open-reasoner", "llama-rl"])
    parser.add_argument("--dataset", type=str, default="math", 
                        choices=["math", "trivia", "mmlu", "math500"])
    parser.add_argument("--num_problems", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ablation_type", type=str, default="layer", 
                        choices=["layer", "head", "layer_range", "permute"])
    parser.add_argument("--start_layer", type=int, default=0)
    parser.add_argument("--end_layer", type=int, default=5)
    parser.add_argument("--head_idx", type=int, default=0)
    parser.add_argument("--layer_idx", type=int, default=0)
    
    # MMLU-specific arguments
    parser.add_argument("--mmlu_question_type", type=str, default=None,
                        choices=["factual", "reasoning"],
                        help="Filter MMLU questions by type: factual (knowledge recall) or reasoning (logical deduction)")
    parser.add_argument("--mmlu_subjects", type=str, nargs="+", default=None,
                        help="Filter MMLU to specific subjects (e.g., --mmlu_subjects college_mathematics high_school_physics)")
    
    # Math500-specific arguments
    parser.add_argument("--math500_categories", type=str, nargs="+", default=None,
                        choices=["Prealgebra", "Algebra", "Number Theory", 
                                "Counting & Probability", "Geometry", 
                                "Intermediate Algebra", "Precalculus"],
                        help="Filter Math500 to specific categories (e.g., --math500_categories Algebra Geometry)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize wandb with simplified run name
    wandb.login(key="e37e9ef98269d14868961ae71509aacef7c8348a")
    
    run_name = f"{args.model}_{args.dataset}_{args.ablation_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    wandb.init(project=f"GGGGGGG", name=run_name)
    
    # Log experiment parameters
    wandb.config.update({
        "model": args.model,
        "dataset": args.dataset,
        "num_problems": args.num_problems,
        "batch_size": args.batch_size,
        "ablation_type": args.ablation_type,
        "start_layer": args.start_layer,
        "end_layer": args.end_layer,
        "head_idx": args.head_idx,
        "layer_idx": args.layer_idx,
        "mmlu_question_type": args.mmlu_question_type,
        "mmlu_subjects": args.mmlu_subjects,
        "math500_categories": args.math500_categories
    })
    
    # Initialize model and dataset
    model = get_model(args.model)
    
    if args.dataset == "math":
        dataset = GSM8KLoader()
        trainer = MathTrainer(
            model, 
            dataset, 
            args.num_problems, 
            args.batch_size,
            model_name=args.model,
            dataset_name="math"
        )
    elif args.dataset == "trivia":
        dataset = TriviaQALoader()
        trainer = TriviaTrainer(
            model, 
            dataset, 
            args.num_problems, 
            args.batch_size,
            model_name=args.model,
            dataset_name="trivia"
        )
    elif args.dataset == "mmlu":
        # Create MMLU dataset with filtering options
        dataset = MMLULoader(
            split="test",
            subject_filter=args.mmlu_subjects,
            question_type_filter=args.mmlu_question_type
        )
        
        # Print dataset statistics
        stats = dataset.get_stats()
        print(f"MMLU Dataset loaded:")
        print(f"  Total questions: {stats['total_questions']}")
        print(f"  Number of subjects: {stats['num_subjects']}")
        print(f"  Question types: {stats['question_types']}")
        print(f"  Subjects: {list(stats['subjects'].keys())}")
        
        trainer = MMLUTrainer(
            model, 
            dataset, 
            args.num_problems, 
            args.batch_size,
            model_name=args.model,
            dataset_name="mmlu"
        )
    elif args.dataset == "math500":
        # Create Math500 dataset with filtering options
        dataset = Math500Loader(
            split="test",  # Math500 only has test split
            category_filter=args.math500_categories
        )
        
        # Print dataset statistics
        stats = dataset.get_stats()
        print(f"Math500 Dataset loaded:")
        print(f"  Total problems: {stats['total_problems']}")
        print(f"  Number of categories: {stats['num_categories']}")
        print(f"  Categories: {list(stats['categories'].keys())}")
        if 'levels' in stats:
            print(f"  Levels: {list(stats['levels'].keys())}")
        
        trainer = Math500Trainer(
            model, 
            dataset, 
            args.num_problems, 
            args.batch_size,
            model_name=args.model,
            dataset_name="math500"
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Run experiment based on ablation type
    if args.ablation_type == "layer":
        results = trainer.run_layer_ablation()
    elif args.ablation_type == "head":
        results = trainer.run_head_ablation(args.layer_idx, args.head_idx)
    elif args.ablation_type == "layer_range":
        results = trainer.run_layer_range_ablation(args.start_layer, args.end_layer)
    elif args.ablation_type == "permute":
        results = trainer.run_permutation(args.start_layer, args.end_layer)
    else:
        raise ValueError(f"Unknown ablation type: {args.ablation_type}")
    
    print(f"\n🎉 Experiment completed!")
    print(f"📁 Results saved in: results/{args.model}-{args.dataset}/")
    print(f"📊 Check the detailed JSON files for questions, responses, and correctness")
    
    wandb.finish()


if __name__ == "__main__":
    main()