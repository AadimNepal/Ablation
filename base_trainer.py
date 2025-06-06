import wandb
import os
import sys
import json
import shutil
from abc import ABC, abstractmethod
from tqdm import tqdm
from datetime import datetime


class BaseTrainer(ABC):
    """Base trainer class with JSON saving functionality"""
    
    def __init__(self, model, dataset, num_problems=50, batch_size=16, model_name="unknown", dataset_name="unknown"):
        self.model = model
        self.dataset = dataset
        self.num_problems = num_problems
        self.batch_size = batch_size
        self.model_name = model_name
        self.dataset_name = dataset_name
        
        # Create timestamp for this experiment run
        self.experiment_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create specific experiment directory for this run
        self.experiment_dir = self._create_experiment_directory()
        print(f"Experiment results will be saved to: {self.experiment_dir}")
    def _create_experiment_directory(self):
        """Create specific experiment directory - can be overridden by subclasses"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dir_name = f"{self.model_name}-{self.dataset_name}-{timestamp}"
        
        experiment_path = os.path.join("results", dir_name)
        os.makedirs(experiment_path, exist_ok=True)
        
        return experiment_path
    
    def create_result_entry(self, item, prompt, response, ground_truth, prediction, is_correct, problem_index):
        """Create a result entry - default implementation, can be overridden by subclasses"""
        return {
            "problem_index": problem_index,
            "prompt": prompt,
            "model_response": response,
            "extracted_prediction": prediction,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "dataset_item": item
        }
    
    def _save_experiment_summary(self, baseline_accuracy, results_summary):
        """Save high-level experiment summary with accuracy per layer"""
        summary_data = {
            "experiment_info": {
                "model_name": self.model_name,
                "dataset_name": self.dataset_name,
                "num_problems": self.num_problems,
                "batch_size": self.batch_size,
                "timestamp": datetime.now().isoformat(),
                "total_layers": self.get_num_layers()
            },
            "baseline_accuracy": baseline_accuracy,
            "layer_results": results_summary
        }
        
        # Save summary to JSON file
        summary_filepath = os.path.join(self.experiment_dir, "experiment_summary.json")
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved experiment summary to {summary_filepath}")
        
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        elif callable(obj):
            return str(obj)
        else:
            return obj
    
    def _prepare_for_json(self, data):
        """Recursively prepare data for JSON serialization"""
        if isinstance(data, dict):
            return {key: self._prepare_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        else:
            return self._make_json_serializable(data)
    
    def _save_layer_results(self, layer_name, results_data):
        """Save results for a specific layer to JSON file"""
        filename = f"{layer_name}.json"
        filepath = os.path.join(self.experiment_dir, filename)
        
        # Create the data structure for this layer
        layer_data = {
            "experiment_info": {
                "model_name": self.model_name,
                "dataset_name": self.dataset_name,
                "layer": layer_name,
                "num_problems": len(results_data),
                "timestamp": datetime.now().isoformat()
            },
            "results": results_data
        }
        
        # Save to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(layer_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(results_data)} results to {filepath}")
        
    def get_num_layers(self):
        """Get number of layers in the model"""

        if hasattr(self.model.hf_model, "transformer") and hasattr(self.model.hf_model.transformer, "h"):
            num_layers = len(self.model.hf_model.transformer.h)
            return num_layers
        elif hasattr(self.model.hf_model, "model") and hasattr(self.model.hf_model.model, "layers"):
            num_layers = len(self.model.hf_model.model.layers)
            return num_layers

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
        """Evaluate batch and save detailed results to JSON"""
        correct, total = 0, 0
        results = []
        
        total_problems = min(self.num_problems, len(self.dataset))
        print(f"Evaluating {layer} - Processing {total_problems} problems...")
        
        batch_count = 0
        for batch_idx in tqdm(range(0, total_problems, self.batch_size), desc=f"Layer {layer} evaluation"):
            batch_count += 1
            
            batch_end_idx = min(batch_idx + self.batch_size, total_problems)
            batch_items = [self.dataset[idx] for idx in range(batch_idx, batch_end_idx)]
            
            # Prepare prompts
            batch_prompts = []
            for item in batch_items:
                prompt = self.prepare_prompt(item)
                batch_prompts.append(prompt)
            
            # Get ground truths
            batch_ground_truths = []
            for item in batch_items:
                ground_truth = self.get_ground_truth(item)
                batch_ground_truths.append(ground_truth)
    
            # Generate responses
            batch_responses = self.model.generate_batch(batch_prompts)
            
            # Process each response and create detailed result entry
            for idx, (item, prompt, response, ground_truth) in enumerate(zip(batch_items, batch_prompts, batch_responses, batch_ground_truths)):
                prediction = self.extract_answer(response)
                is_correct = self.check_correctness(response, ground_truth)
                
                if is_correct:
                    correct += 1
                total += 1
                
                # Create result entry (overridden by subclasses for dataset-specific fields)
                result_entry = self.create_result_entry(
                    item, prompt, response, ground_truth, prediction, is_correct, batch_idx + idx
                )
                
                results.append(result_entry)

                # Log to wandb
                try:
                    wandb.log({
                        f"layer_{layer}_accuracy": correct/total,
                        f"layer_{layer}_correct": int(is_correct)
                    })
                except Exception as e:
                    print(f"wandb.log failed: {e}")
        
        final_accuracy = correct / total
        print(f"Layer {layer} accuracy: {final_accuracy:.4f} ({correct}/{total})")
        
        # Save results to JSON file
        self._save_layer_results(layer, results)
        
        return final_accuracy

    def run_layer_ablation(self):
        """Run layer-by-layer ablation experiment with JSON saving"""
        print(f"Starting layer ablation experiment for {self.model_name} on {self.dataset_name}")
        print(f"Results will be saved to: {self.experiment_dir}")
        
        print("Evaluating baseline model...")
        baseline_accuracy = self.evaluate_batch("baseline")
        
        try:
            wandb.log({"layer_accuracy": baseline_accuracy, "layer_number": "baseline"})
        except Exception as e:
            print(f"wandb baseline log failed: {e}")
        
        results_summary = []
        num_layers = self.get_num_layers()
        print(f"Model has {num_layers} layers")
        
        # Run layer-by-layer ablation
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                # Special case for layer 0 - just log 0 accuracy
                try:
                    wandb.log({
                        "layer_accuracy": 0,
                        "layer_number": 0
                    })
                except Exception as e:
                    print(f"wandb layer 0 log failed: {e}")
                
                results_summary.append({
                    "layer": 0,
                    "accuracy": 0,
                    "delta": baseline_accuracy,
                    "note": "Layer 0 assumed to have 0 accuracy"
                })
                continue
            
            print(f"Ablating layer {layer_idx}...")
            
            self.model.zero_ablate(layer_idx)
            accuracy = self.evaluate_batch(layer=f"layer_{layer_idx}")
            delta = baseline_accuracy - accuracy
            
            try:
                wandb.log({
                    "layer_accuracy": accuracy,
                    "layer_number": layer_idx
                })
            except Exception as e:
                print(f"wandb layer {layer_idx} log failed: {e}")
            
            results_summary.append({
                "layer": layer_idx,
                "accuracy": accuracy,
                "delta": delta
            })
            
            print(f"Layer {layer_idx}: {accuracy:.4f} (Δ {delta:.4f})")
        
        print(f"\nExperiment completed!")
        print(f"Results saved in: {self.experiment_dir}")
        
        # Save high-level experiment summary
        self._save_experiment_summary(baseline_accuracy, results_summary)
        
        # Clean up ablated model checkpoints
        self._cleanup_ablated_models()
        
        return {"baseline": baseline_accuracy, "results": results_summary}
    
    def _cleanup_ablated_models(self):
        """Clean up ablated model checkpoint directories after experiment"""
        try:
            if hasattr(self.model, 'checkpoint_path') and os.path.exists(self.model.checkpoint_path):
                print(f"Cleaning up ablated model checkpoint: {self.model.checkpoint_path}")
                shutil.rmtree(self.model.checkpoint_path)
                print("✓ Ablated model checkpoint cleaned up")
        except Exception as e:
            print(f"Warning: Failed to clean up checkpoint directory: {e}")