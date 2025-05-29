import wandb
import os
import json
from abc import ABC, abstractmethod
from tqdm import tqdm


class BaseTrainer(ABC):
    """Base trainer class with common functionality"""
    
    def __init__(self, model, dataset, num_problems=50, batch_size=16):
        self.model = model
        self.dataset = dataset
        self.num_problems = num_problems
        self.batch_size = batch_size
        
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
        """Evaluate model on dataset using batched processing"""
        correct, total = 0, 0
        results = []
        
        total_problems = min(self.num_problems, len(self.dataset))
        
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
            for idx, (response, ground_truth) in enumerate(zip(batch_responses, batch_ground_truths)):
                prediction = self.extract_answer(response)
                is_correct = self.check_correctness(prediction, ground_truth)
                
                if is_correct:
                    correct += 1
                total += 1
                
                results.append({
                    "index": batch_idx + idx,
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                    "is_correct": is_correct,
                    "response": response
                })
                
                # Log progress
                wandb.log({
                    f"layer_{layer}_accuracy": correct/total,
                    f"layer_{layer}_correct": int(is_correct)
                })
        
        # Save results
        os.makedirs("results", exist_ok=True)
        with open(f"results/layer_{layer}_results.json", "w") as f:
            json.dump({"results": results}, f, indent=2)
        
        return correct / total
    
    def run_layer_ablation(self):
        """Run layer-by-layer ablation experiment"""
        print("Evaluating baseline model...")
        baseline_accuracy = self.evaluate_batch("baseline")
        wandb.log({"layer_accuracy": baseline_accuracy, "layer_number": "baseline"})
        print(f"Baseline accuracy: {baseline_accuracy:.4f}")
        
        results = []
        num_layers = self.get_num_layers()
        print(f"Model has {num_layers} layers")
        
        for layer_idx in range(num_layers):

            if layer_idx == 0:

                wandb.log({
                    "layer_accuracy": 0,
                    "layer_number": 0
                })

                results.append({"layer": 0, "accuracy": 0, "delta": 0})
                continue

            print(f"Ablating layer {layer_idx}...")
            self.model.zero_ablate(layer_idx)
            
            accuracy = self.evaluate_batch(layer=layer_idx)
            delta = baseline_accuracy - accuracy
            
            wandb.log({
                "layer_accuracy": accuracy,
                "layer_number": layer_idx
            })
            
            results.append({"layer": layer_idx, "accuracy": accuracy, "delta": delta})
            print(f"Layer {layer_idx}: {accuracy:.4f} (Δ {delta:.4f})")
        
        return {"baseline": baseline_accuracy, "results": results}
    
    def run_head_ablation(self, layer_idx=0, head_idx=0):
        """Run head ablation experiment"""
        print("Evaluating baseline model...")
        baseline_accuracy = self.evaluate_batch("baseline")
        wandb.log({"baseline_accuracy": baseline_accuracy})
        print(f"Baseline accuracy: {baseline_accuracy:.4f}")
        
        print(f"Ablating head {head_idx} in layer {layer_idx}...")
        self.model.zero_ablate_head(layer_idx, head_idx)
        accuracy = self.evaluate_batch(layer=f"{layer_idx}_head_{head_idx}")
        delta = baseline_accuracy - accuracy
        
        wandb.log({
            "head_accuracy": accuracy,
            "layer_number": layer_idx,
            "head_number": head_idx
        })
        
        print(f"Head {head_idx} in layer {layer_idx}: {accuracy:.4f} (Δ {delta:.4f})")
        return {
            "baseline": baseline_accuracy, 
            "head_result": {
                "layer": layer_idx, 
                "head": head_idx, 
                "accuracy": accuracy, 
                "delta": delta
            }
        }
    
    def run_layer_range_ablation(self, start_layer=0, end_layer=5):
        """Run layer range ablation experiment"""
        print("Evaluating baseline model...")
        baseline_accuracy = self.evaluate_batch("baseline")
        wandb.log({"baseline_accuracy": baseline_accuracy})
        print(f"Baseline accuracy: {baseline_accuracy:.4f}")
        
        print(f"Ablating layers {start_layer} to {end_layer-1}...")
        self.model.ablate_layers_range(start_layer, end_layer)
        accuracy = self.evaluate_batch(layer=f"range_{start_layer}_{end_layer}")
        delta = baseline_accuracy - accuracy
        
        wandb.log({
            "range_accuracy": accuracy,
            "start_layer": start_layer,
            "end_layer": end_layer
        })
        
        print(f"Layers {start_layer}-{end_layer-1}: {accuracy:.4f} (Δ {delta:.4f})")
        return {
            "baseline": baseline_accuracy, 
            "range_result": {
                "start": start_layer, 
                "end": end_layer, 
                "accuracy": accuracy, 
                "delta": delta
            }
        }
    
    def run_permutation(self, start_layer=0, end_layer=5):
        """Run layer permutation experiment"""
        print("Evaluating baseline model...")
        baseline_accuracy = self.evaluate_batch("baseline")
        wandb.log({"baseline_accuracy": baseline_accuracy})
        print(f"Baseline accuracy: {baseline_accuracy:.4f}")
        
        print(f"Permuting layers {start_layer} to {end_layer-1}...")
        permutation_result = self.model.permute_layers(start_layer, end_layer)
        print(f"Permutation applied: {permutation_result}")
        
        accuracy = self.evaluate_batch(layer=f"permute_{start_layer}_{end_layer}")
        delta = baseline_accuracy - accuracy
        
        wandb.log({
            "permute_accuracy": accuracy,
            "start_layer": start_layer,
            "end_layer": end_layer
        })
        
        print(f"Permuted layers {start_layer}-{end_layer-1}: {accuracy:.4f} (Δ {delta:.4f})")
        return {
            "baseline": baseline_accuracy, 
            "permute_result": {
                "start": start_layer, 
                "end": end_layer, 
                "accuracy": accuracy, 
                "delta": delta, 
                "permutation": permutation_result
            }
        }