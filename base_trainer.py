import wandb
import os
import json
from abc import ABC, abstractmethod
from tqdm import tqdm
from datetime import datetime
import random
import csv


class BaseTrainer(ABC):
    """Base trainer class with improved result organization"""
    
    def __init__(self, model, dataset, num_problems=50, batch_size=16, model_name="unknown", dataset_name="unknown"):
        self.model = model
        self.dataset = dataset
        self.num_problems = num_problems
        self.batch_size = batch_size
        self.model_name = model_name
        self.dataset_name = dataset_name
        
        # Create organized results directory
        self.results_dir = f"results/{model_name}-{dataset_name}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create timestamp for this experiment run
        self.experiment_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def get_num_layers(self):
        """Get number of layers in the model"""
        if hasattr(self.model.hf_model, "transformer") and hasattr(self.model.hf_model.transformer, "h"):
            return len(self.model.hf_model.transformer.h) 
        elif hasattr(self.model.hf_model, "model") and hasattr(self.model.hf_model.model, "layers"):
            return len(self.model.hf_model.model.layers)  
        return 32
    
    def get_num_heads(self):
        """Get number of attention heads in the model"""
        if hasattr(self.model.hf_model, "config"):
            if hasattr(self.model.hf_model.config, "num_attention_heads"):
                return self.model.hf_model.config.num_attention_heads
            elif hasattr(self.model.hf_model.config, "n_head"):
                return self.model.hf_model.config.n_head
        return 32
    
    @abstractmethod
    def extract_answer(self, output):
        """Extract answer from model output"""
        pass
    
    @abstractmethod
    def prepare_prompt(self, item):
        """Prepare prompt for the model"""
        pass
    
    @abstractmethod
    def get_ground_truth(self, item):
        """Get ground truth answer from dataset item"""
        pass
    
    @abstractmethod
    def check_correctness(self, prediction, ground_truth):
        """Check if prediction matches ground truth"""
        pass

    def save_spreadsheet(self, results, layer):
        """Save results as CSV spreadsheet"""
        csv_filename = f"layer_{layer}_results.csv"
        csv_filepath = os.path.join(self.results_dir, csv_filename)
        
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'question', 
                'model_response', 
                'extracted_answer', 
                'correct_answer', 
                'model_name',
                'is_correct'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                writer.writerow({
                    'question': result['prompt'],
                    'model_response': result['model_response'],
                    'extracted_answer': result['extracted_prediction'],
                    'correct_answer': result['ground_truth'],
                    'model_name': self.model_name,
                    'is_correct': 1 if result['is_correct'] else 0
                })
        
        print(f"Spreadsheet saved to: {csv_filepath}")

    def evaluate_batch(self, layer="baseline"):
        """Evaluate model on dataset using batched processing with detailed saving"""
        correct, total = 0, 0
        results = []
        
        dataset_size = len(self.dataset)
        requested_problems = self.num_problems
        
        # Handle edge cases
        if dataset_size == 0:
            print("Warning: Dataset is empty!")
            return 0.0
        
        if requested_problems <= 0:
            print("Warning: Requested 0 or negative problems!")
            return 0.0
        
        # Determine how many problems to actually process
        total_problems = min(requested_problems, dataset_size)
        
        # Generate random indices for sampling
        if total_problems == dataset_size:
            # Use all samples (either requested all, or requested more than available)
            print(f"Using all {dataset_size} samples from dataset")
            indices = list(range(dataset_size))
            random.shuffle(indices)  # Still randomize the order
        else:
            # Sample random subset without replacement
            print(f"Sampling {total_problems} random problems from {dataset_size} available")
            indices = random.sample(range(dataset_size), total_problems)
        
        print(f"Evaluating {layer} - Processing {total_problems} problems...")
        
        # Process in batches using the selected indices
        for batch_idx in tqdm(range(0, total_problems, self.batch_size), desc=f"Layer {layer} evaluation"):
            batch_end_idx = min(batch_idx + self.batch_size, total_problems)
            
            # Get items using selected indices (guaranteed to be within bounds)
            batch_indices = indices[batch_idx:batch_end_idx]
            batch_items = [self.dataset[idx] for idx in batch_indices]
            
            # Prepare prompts and ground truths
            batch_prompts = [self.prepare_prompt(item) for item in batch_items]
            batch_ground_truths = [self.get_ground_truth(item) for item in batch_items]
            
            # Generate responses using generate_batch
            batch_responses = self.model.generate_batch(batch_prompts)
            
            # Process each response
            for idx, (item, prompt, response, ground_truth, dataset_idx) in enumerate(zip(batch_items, batch_prompts, batch_responses, batch_ground_truths, batch_indices)):
                prediction = self.extract_answer(response)
                is_correct = self.check_correctness(prediction, ground_truth)
                
                if is_correct:
                    correct += 1
                total += 1
                
                # Store detailed results including the original question/prompt
                result_entry = {
                    "index": batch_idx + idx,
                    "dataset_index": dataset_idx,
                    "prompt": prompt,
                    "ground_truth": ground_truth,
                    "model_response": response,
                    "extracted_prediction": prediction,
                    "is_correct": is_correct,
                    "dataset_item": item
                }
                
                results.append(result_entry)
                
                # Log progress to wandb
                wandb.log({
                    f"layer_{layer}_accuracy": correct/total,
                    f"layer_{layer}_correct": int(is_correct)
                })
        
        # Save detailed results for this layer
        layer_filename = f"layer_{layer}_detailed.json"
        layer_filepath = os.path.join(self.results_dir, layer_filename)
        
        layer_data = {
            "experiment_info": {
                "model": self.model_name,
                "dataset": self.dataset_name,
                "layer": layer,
                "timestamp": self.experiment_timestamp,
                "total_problems": total_problems,
                "batch_size": self.batch_size,
                "random_indices": indices,
                "dataset_size": dataset_size
            },
            "summary": {
                "total_examples": total,
                "correct_examples": correct,
                "accuracy": correct / total,
                "incorrect_examples": total - correct
            },
            "detailed_results": results
        }
        
        with open(layer_filepath, "w") as f:
            json.dump(layer_data, f, indent=2)
        
        # Save as spreadsheet
        self.save_spreadsheet(results, layer)
        
        print(f"Layer {layer} results saved to: {layer_filepath}")
        print(f"Layer {layer} accuracy: {correct/total:.4f} ({correct}/{total})")
        
        return correct / total
    
    def run_layer_ablation(self):
        """Run layer-by-layer ablation experiment with organized results"""
        print(f"Starting layer ablation experiment for {self.model_name} on {self.dataset_name}")
        print(f"Results will be saved to: {self.results_dir}")
        
        # Evaluate baseline model
        print("Evaluating baseline model...")
        baseline_accuracy = self.evaluate_batch("baseline")
        wandb.log({"layer_accuracy": baseline_accuracy, "layer_number": "baseline"})
        
        results_summary = []
        num_layers = self.get_num_layers()
        print(f"Model has {num_layers} layers")
        
        # Run layer-by-layer ablation
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                # Special case for layer 0 - just log 0 accuracy
                wandb.log({
                    "layer_accuracy": 0,
                    "layer_number": 0
                })
                results_summary.append({
                    "layer": 0, 
                    "accuracy": 0, 
                    "delta": baseline_accuracy,
                    "note": "Layer 0 assumed to have 0 accuracy"
                })
                continue

            print(f"Ablating layer {layer_idx}...")
            self.model.zero_ablate(layer_idx)
            
            accuracy = self.evaluate_batch(layer=layer_idx)
            delta = baseline_accuracy - accuracy
            
            wandb.log({
                "layer_accuracy": accuracy,
                "layer_number": layer_idx
            })
            
            results_summary.append({
                "layer": layer_idx, 
                "accuracy": accuracy, 
                "delta": delta
            })
            
            print(f"Layer {layer_idx}: {accuracy:.4f} (Δ {delta:.4f})")
        
        # Save overall experiment summary
        experiment_summary = {
            "experiment_info": {
                "model": self.model_name,
                "dataset": self.dataset_name,
                "experiment_type": "layer_ablation",
                "timestamp": self.experiment_timestamp,
                "total_problems": self.num_problems,
                "batch_size": self.batch_size,
                "num_layers": num_layers
            },
            "baseline_accuracy": baseline_accuracy,
            "layer_results": results_summary,
            "files_generated": [
                f"layer_baseline_detailed.json",
                *[f"layer_{i}_detailed.json" for i in range(num_layers) if i != 0]
            ]
        }
        
        summary_filepath = os.path.join(self.results_dir, f"experiment_summary_{self.experiment_timestamp}.json")
        with open(summary_filepath, "w") as f:
            json.dump(experiment_summary, f, indent=2)
        
        print(f"\nExperiment completed!")
        print(f"Summary saved to: {summary_filepath}")
        print(f"All detailed results in: {self.results_dir}")
        
        return {"baseline": baseline_accuracy, "results": results_summary}