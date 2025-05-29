import wandb
import os
import json
from abc import ABC, abstractmethod
from tqdm import tqdm
from datetime import datetime


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
    
    def evaluate_batch(self, layer="baseline"):
        """Evaluate model on dataset using batched processing with detailed saving"""
        correct, total = 0, 0
        results = []
        
        total_problems = min(self.num_problems, len(self.dataset))
        
        print(f"Evaluating {layer} - Processing {total_problems} problems...")
        
        # Process in batches
        for batch_idx in tqdm(range(0, total_problems, self.batch_size), desc=f"Layer {layer} evaluation"):
            batch_end_idx = min(batch_idx + self.batch_size, total_problems)
            batch_items = [self.dataset[idx] for idx in range(batch_idx, batch_end_idx)]
            
            # Prepare prompts and ground truths
            batch_prompts = [self.prepare_prompt(item) for item in batch_items]
            batch_ground_truths = [self.get_ground_truth(item) for item in batch_items]
            
            # Generate responses using generate_batch
            batch_responses = self.model.generate_batch(batch_prompts)
            
            # Process each response
            for idx, (item, prompt, response, ground_truth) in enumerate(zip(batch_items, batch_prompts, batch_responses, batch_ground_truths)):
                prediction = self.extract_answer(response)
                is_correct = self.check_correctness(prediction, ground_truth)
                
                if is_correct:
                    correct += 1
                total += 1
                
                # Store detailed results including the original question/prompt
                result_entry = {
                    "index": batch_idx + idx,
                    "prompt": prompt,  # The question/input
                    "ground_truth": ground_truth,
                    "model_response": response,  # Full model output
                    "extracted_prediction": prediction,  # What we extracted as the answer
                    "is_correct": is_correct,
                    "dataset_item": item  # Store original dataset item for reference
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
                "batch_size": self.batch_size
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
    
    def run_head_ablation(self, layer_idx=0, head_idx=0):
        """Run head ablation experiment with organized results"""
        print(f"Starting head ablation experiment for {self.model_name} on {self.dataset_name}")
        print(f"Ablating head {head_idx} in layer {layer_idx}")
        
        # Evaluate baseline
        baseline_accuracy = self.evaluate_batch("baseline")
        wandb.log({"baseline_accuracy": baseline_accuracy})
        
        # Ablate specific head
        self.model.zero_ablate_head(layer_idx, head_idx)
        head_label = f"{layer_idx}_head_{head_idx}"
        accuracy = self.evaluate_batch(layer=head_label)
        delta = baseline_accuracy - accuracy
        
        wandb.log({
            "head_accuracy": accuracy,
            "layer_number": layer_idx,
            "head_number": head_idx
        })
        
        # Save experiment summary
        experiment_summary = {
            "experiment_info": {
                "model": self.model_name,
                "dataset": self.dataset_name,
                "experiment_type": "head_ablation",
                "timestamp": self.experiment_timestamp,
                "layer_idx": layer_idx,
                "head_idx": head_idx
            },
            "baseline_accuracy": baseline_accuracy,
            "head_result": {
                "layer": layer_idx, 
                "head": head_idx, 
                "accuracy": accuracy, 
                "delta": delta
            }
        }
        
        summary_filepath = os.path.join(self.results_dir, f"head_ablation_summary_{self.experiment_timestamp}.json")
        with open(summary_filepath, "w") as f:
            json.dump(experiment_summary, f, indent=2)
        
        print(f"Head ablation completed! Summary: {summary_filepath}")
        return experiment_summary
    
    def run_layer_range_ablation(self, start_layer=0, end_layer=5):
        """Run layer range ablation experiment with organized results"""
        print(f"Starting layer range ablation for {self.model_name} on {self.dataset_name}")
        print(f"Ablating layers {start_layer} to {end_layer-1}")
        
        # Evaluate baseline
        baseline_accuracy = self.evaluate_batch("baseline")
        wandb.log({"baseline_accuracy": baseline_accuracy})
        
        # Ablate layer range
        self.model.ablate_layers_range(start_layer, end_layer)
        range_label = f"range_{start_layer}_{end_layer}"
        accuracy = self.evaluate_batch(layer=range_label)
        delta = baseline_accuracy - accuracy
        
        wandb.log({
            "range_accuracy": accuracy,
            "start_layer": start_layer,
            "end_layer": end_layer
        })
        
        # Save experiment summary
        experiment_summary = {
            "experiment_info": {
                "model": self.model_name,
                "dataset": self.dataset_name,
                "experiment_type": "layer_range_ablation",
                "timestamp": self.experiment_timestamp,
                "start_layer": start_layer,
                "end_layer": end_layer
            },
            "baseline_accuracy": baseline_accuracy,
            "range_result": {
                "start": start_layer, 
                "end": end_layer, 
                "accuracy": accuracy, 
                "delta": delta
            }
        }
        
        summary_filepath = os.path.join(self.results_dir, f"range_ablation_summary_{self.experiment_timestamp}.json")
        with open(summary_filepath, "w") as f:
            json.dump(experiment_summary, f, indent=2)
        
        print(f"Range ablation completed! Summary: {summary_filepath}")
        return experiment_summary
    
    def run_permutation(self, start_layer=0, end_layer=5):
        """Run layer permutation experiment with organized results"""
        print(f"Starting layer permutation for {self.model_name} on {self.dataset_name}")
        print(f"Permuting layers {start_layer} to {end_layer-1}")
        
        # Evaluate baseline
        baseline_accuracy = self.evaluate_batch("baseline")
        wandb.log({"baseline_accuracy": baseline_accuracy})
        
        # Apply permutation
        permutation_result = self.model.permute_layers(start_layer, end_layer)
        permute_label = f"permute_{start_layer}_{end_layer}"
        accuracy = self.evaluate_batch(layer=permute_label)
        delta = baseline_accuracy - accuracy
        
        wandb.log({
            "permute_accuracy": accuracy,
            "start_layer": start_layer,
            "end_layer": end_layer
        })
        
        # Save experiment summary
        experiment_summary = {
            "experiment_info": {
                "model": self.model_name,
                "dataset": self.dataset_name,
                "experiment_type": "layer_permutation",
                "timestamp": self.experiment_timestamp,
                "start_layer": start_layer,
                "end_layer": end_layer
            },
            "baseline_accuracy": baseline_accuracy,
            "permute_result": {
                "start": start_layer, 
                "end": end_layer, 
                "accuracy": accuracy, 
                "delta": delta, 
                "permutation": permutation_result
            }
        }
        
        summary_filepath = os.path.join(self.results_dir, f"permutation_summary_{self.experiment_timestamp}.json")
        with open(summary_filepath, "w") as f:
            json.dump(experiment_summary, f, indent=2)
        
        print(f"Permutation completed! Summary: {summary_filepath}")
        return experiment_summary