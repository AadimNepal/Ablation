import wandb
import os
import sys
from abc import ABC, abstractmethod
from tqdm import tqdm
from datetime import datetime


class BaseTrainer(ABC):
    """Base trainer class with extensive debugging - NO JSON SAVES"""
    
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
        
        print(f"DEBUG: BaseTrainer initialized")
        print(f"DEBUG: Results directory: {self.results_dir}")
        print(f"DEBUG: Experiment timestamp: {self.experiment_timestamp}")
        print(f"DEBUG: NO JSON FILES WILL BE SAVED - PURE DEBUG MODE")
        
    def get_num_layers(self):
        """Get number of layers in the model"""
        print(f"DEBUG: Getting number of layers...")
        if hasattr(self.model.hf_model, "transformer") and hasattr(self.model.hf_model.transformer, "h"):
            num_layers = len(self.model.hf_model.transformer.h)
            print(f"DEBUG: Found {num_layers} layers in transformer.h")
            return num_layers
        elif hasattr(self.model.hf_model, "model") and hasattr(self.model.hf_model.model, "layers"):
            num_layers = len(self.model.hf_model.model.layers)
            print(f"DEBUG: Found {num_layers} layers in model.layers")
            return num_layers
        print(f"DEBUG: Could not determine layers, defaulting to 32")
        return 32
    
    def get_num_heads(self):
        """Get number of attention heads in the model"""
        print(f"DEBUG: Getting number of attention heads...")
        if hasattr(self.model.hf_model, "config"):
            if hasattr(self.model.hf_model.config, "num_attention_heads"):
                num_heads = self.model.hf_model.config.num_attention_heads
                print(f"DEBUG: Found {num_heads} attention heads")
                return num_heads
            elif hasattr(self.model.hf_model.config, "n_head"):
                num_heads = self.model.hf_model.config.n_head
                print(f"DEBUG: Found {num_heads} heads (n_head)")
                return num_heads
        print(f"DEBUG: Could not determine heads, defaulting to 32")
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
        """Evaluate model on dataset - NO FILE SAVING, PURE DEBUG"""
        print(f"\n" + "="*80)
        print(f"DEBUG: Starting evaluate_batch for layer={layer}")
        print(f"DEBUG: Current time: {datetime.now()}")
        print(f"DEBUG: num_problems: {self.num_problems}, dataset length: {len(self.dataset)}")
        print(f"DEBUG: batch_size: {self.batch_size}")
        
        correct, total = 0, 0
        results = []
        
        total_problems = min(self.num_problems, len(self.dataset))
        print(f"DEBUG: Will process {total_problems} problems")
        
        print(f"Evaluating {layer} - Processing {total_problems} problems...")
        
        # Process in batches
        batch_count = 0
        for batch_idx in tqdm(range(0, total_problems, self.batch_size), desc=f"Layer {layer} evaluation"):
            batch_count += 1
            print(f"\nDEBUG: === BATCH {batch_count} START ===")
            print(f"DEBUG: Batch {batch_count} starting at index {batch_idx}")
            print(f"DEBUG: Current time: {datetime.now()}")
            
            batch_end_idx = min(batch_idx + self.batch_size, total_problems)
            print(f"DEBUG: Batch range: {batch_idx} to {batch_end_idx}")
            
            try:
                print(f"DEBUG: Getting batch items from dataset...")
                batch_items = [self.dataset[idx] for idx in range(batch_idx, batch_end_idx)]
                print(f"DEBUG: Successfully got {len(batch_items)} batch items")
                
                # Check first item structure
                if batch_count == 1 and len(batch_items) > 0:
                    first_item = batch_items[0]
                    print(f"DEBUG: First item type: {type(first_item)}")
                    print(f"DEBUG: First item size: {sys.getsizeof(first_item)} bytes")
                    if hasattr(first_item, 'keys'):
                        print(f"DEBUG: First item keys: {list(first_item.keys())}")
                    else:
                        print(f"DEBUG: First item has no keys attribute")
                        
            except Exception as e:
                print(f"DEBUG ERROR: Failed to get batch items: {e}")
                print(f"DEBUG: Exception type: {type(e)}")
                raise
            
            # Prepare prompts and ground truths
            try:
                print(f"DEBUG: Preparing prompts for {len(batch_items)} items...")
                batch_prompts = []
                for i, item in enumerate(batch_items):
                    if i == 0:
                        print(f"DEBUG: Preparing prompt for first item...")
                    prompt = self.prepare_prompt(item)
                    batch_prompts.append(prompt)
                    if i == 0:
                        print(f"DEBUG: First prompt length: {len(str(prompt))} chars")
                        print(f"DEBUG: First prompt preview: {str(prompt)[:100]}...")
                
                print(f"DEBUG: Prepared {len(batch_prompts)} prompts")
                
                print(f"DEBUG: Getting ground truths...")
                batch_ground_truths = []
                for i, item in enumerate(batch_items):
                    ground_truth = self.get_ground_truth(item)
                    batch_ground_truths.append(ground_truth)
                    if i == 0:
                        print(f"DEBUG: First ground truth: {str(ground_truth)}")
                        
                print(f"DEBUG: Got {len(batch_ground_truths)} ground truths")
                
            except Exception as e:
                print(f"DEBUG ERROR: Failed to prepare prompts/ground truths: {e}")
                print(f"DEBUG: Exception type: {type(e)}")
                raise
            
            # Generate responses using generate_batch
            try:
                print(f"DEBUG: Starting model.generate_batch with {len(batch_prompts)} prompts...")
                print(f"DEBUG: Current time: {datetime.now()}")
                
                batch_responses = self.model.generate_batch(batch_prompts)
                
                print(f"DEBUG: Model generation completed!")
                print(f"DEBUG: Current time: {datetime.now()}")
                print(f"DEBUG: Got {len(batch_responses)} responses from model")
                
                if len(batch_responses) > 0:
                    first_response = batch_responses[0]
                    print(f"DEBUG: First response length: {len(str(first_response))} chars")
                    print(f"DEBUG: First response preview: {str(first_response)[:100]}...")
                    
            except Exception as e:
                print(f"DEBUG ERROR: Failed during model generation: {e}")
                print(f"DEBUG: Exception type: {type(e)}")
                raise
            
            # Process each response
            print(f"DEBUG: Processing {len(batch_responses)} individual responses...")
            for idx, (item, prompt, response, ground_truth) in enumerate(zip(batch_items, batch_prompts, batch_responses, batch_ground_truths)):
                try:
                    if idx == 0:  # Debug first item in detail
                        print(f"DEBUG: Processing first item in batch...")
                        print(f"DEBUG: Item {idx} - prompt length: {len(str(prompt))}")
                        print(f"DEBUG: Response length: {len(str(response))}")
                        print(f"DEBUG: Ground truth: {str(ground_truth)}")
                    
                    print(f"DEBUG: Extracting answer from response...")

                    prediction = self.extract_answer(response)
                    
                    if idx == 0:
                        print(f"DEBUG: Extracted prediction: {str(prediction)}")
                    
                    print(f"DEBUG: Checking correctness...")
                    is_correct = self.check_correctness(response, ground_truth)
                    
                    if idx == 0:
                        print(f"DEBUG: Is correct: {is_correct}")
                    
                    if is_correct:
                        correct += 1
                    total += 1
                    
                    # Store minimal results (no heavy objects)
                    result_entry = {
                        "index": batch_idx + idx,
                        "is_correct": is_correct,
                        "prediction": str(prediction)[:100],  # Truncated
                        "ground_truth": str(ground_truth)[:100],  # Truncated
                        # NO dataset_item stored!
                        # NO full prompt/response stored!
                    }
                    
                    results.append(result_entry)
                    
                    # Log progress to wandb
                    try:
                        wandb.log({
                            f"layer_{layer}_accuracy": correct/total,
                            f"layer_{layer}_correct": int(is_correct)
                        })
                        if idx == 0:
                            print(f"DEBUG: wandb.log successful")
                    except Exception as e:
                        print(f"DEBUG WARNING: wandb.log failed: {e}")
                        
                except Exception as e:
                    print(f"DEBUG ERROR: Failed processing item {idx}: {e}")
                    print(f"DEBUG: Exception type: {type(e)}")
                    raise
            
            print(f"DEBUG: Batch {batch_count} completed successfully")
            print(f"DEBUG: Current accuracy: {correct/total:.4f} ({correct}/{total})")
            print(f"DEBUG: === BATCH {batch_count} END ===")
        
        print(f"\n" + "="*80)
        print(f"DEBUG: Finished processing all batches")
        print(f"DEBUG: Total results collected: {len(results)}")
        print(f"DEBUG: Results list size: {sys.getsizeof(results)} bytes")
        print(f"DEBUG: Current time: {datetime.now()}")
        
        # NO FILE SAVING - Just print summary
        print(f"DEBUG: *** NO JSON FILE SAVING - DEBUG MODE ***")
        print(f"DEBUG: Would have saved to: layer_{layer}_detailed.json")
        print(f"DEBUG: Summary data created successfully")
        
        # Final results
        final_accuracy = correct / total
        print(f"Layer {layer} accuracy: {final_accuracy:.4f} ({correct}/{total})")
        print(f"DEBUG: evaluate_batch completed for layer {layer}")
        print(f"DEBUG: Final time: {datetime.now()}")
        print("="*80)
        
        return final_accuracy
    
    def run_layer_ablation(self):
        """Run layer-by-layer ablation experiment - NO FILE SAVING"""
        print(f"DEBUG: Starting run_layer_ablation")
        print(f"DEBUG: *** NO FILE SAVING MODE - PURE DEBUG ***")
        print(f"Starting layer ablation experiment for {self.model_name} on {self.dataset_name}")
        print(f"Results directory (not used): {self.results_dir}")
        
        # Evaluate baseline model
        print("DEBUG: About to evaluate baseline model...")
        print("Evaluating baseline model...")
        baseline_accuracy = self.evaluate_batch("baseline")
        print(f"DEBUG: Baseline accuracy: {baseline_accuracy}")
        
        try:
            wandb.log({"layer_accuracy": baseline_accuracy, "layer_number": "baseline"})
            print(f"DEBUG: wandb baseline log successful")
        except Exception as e:
            print(f"DEBUG WARNING: wandb baseline log failed: {e}")
        
        results_summary = []
        num_layers = self.get_num_layers()
        print(f"Model has {num_layers} layers")
        
        # Run layer-by-layer ablation
        for layer_idx in range(num_layers):
            print(f"\nDEBUG: ===== LAYER {layer_idx} ABLATION START =====")
            print(f"DEBUG: Current time: {datetime.now()}")
            
            if layer_idx == 0:
                print(f"DEBUG: Special case for layer 0")
                # Special case for layer 0 - just log 0 accuracy
                try:
                    wandb.log({
                        "layer_accuracy": 0,
                        "layer_number": 0
                    })
                    print(f"DEBUG: wandb layer 0 log successful")
                except Exception as e:
                    print(f"DEBUG WARNING: wandb layer 0 log failed: {e}")
                    
                results_summary.append({
                    "layer": 0, 
                    "accuracy": 0, 
                    "delta": baseline_accuracy,
                    "note": "Layer 0 assumed to have 0 accuracy"
                })
                print(f"DEBUG: Layer 0 completed (skipped)")
                continue

            print(f"Ablating layer {layer_idx}...")
            print(f"DEBUG: Calling model.zero_ablate({layer_idx})")
            print(f"DEBUG: Current time: {datetime.now()}")
            
            try:
                self.model.zero_ablate(layer_idx)
                print(f"DEBUG: zero_ablate completed for layer {layer_idx}")
                print(f"DEBUG: Current time: {datetime.now()}")
            except Exception as e:
                print(f"DEBUG ERROR: zero_ablate failed for layer {layer_idx}: {e}")
                print(f"DEBUG: Exception type: {type(e)}")
                raise
            
            print(f"DEBUG: About to evaluate layer {layer_idx}")
            print(f"DEBUG: Current time: {datetime.now()}")
            
            try:
                accuracy = self.evaluate_batch(layer=layer_idx)
                print(f"DEBUG: Layer {layer_idx} evaluation completed")
                print(f"DEBUG: Layer {layer_idx} accuracy: {accuracy}")
                print(f"DEBUG: Current time: {datetime.now()}")
            except Exception as e:
                print(f"DEBUG ERROR: evaluate_batch failed for layer {layer_idx}: {e}")
                print(f"DEBUG: Exception type: {type(e)}")
                raise
            
            delta = baseline_accuracy - accuracy
            
            try:
                wandb.log({
                    "layer_accuracy": accuracy,
                    "layer_number": layer_idx
                })
                print(f"DEBUG: wandb layer {layer_idx} log successful")
            except Exception as e:
                print(f"DEBUG WARNING: wandb layer {layer_idx} log failed: {e}")
            
            results_summary.append({
                "layer": layer_idx, 
                "accuracy": accuracy, 
                "delta": delta
            })
            
            print(f"Layer {layer_idx}: {accuracy:.4f} (Δ {delta:.4f})")
            print(f"DEBUG: ===== LAYER {layer_idx} ABLATION END =====")
        
        print(f"\nDEBUG: All layer ablations completed")
        print(f"DEBUG: *** NO SUMMARY FILE SAVING - DEBUG MODE ***")
        print(f"DEBUG: Would have saved experiment summary")
        
        print(f"\nExperiment completed!")
        print(f"DEBUG: Total layers processed: {len(results_summary)}")
        print(f"Results NOT saved (debug mode)")
        
        return {"baseline": baseline_accuracy, "results": results_summary}
    
    def run_head_ablation(self, layer_idx=0, head_idx=0):
        """Run head ablation experiment - NO FILE SAVING"""
        print(f"DEBUG: Starting head ablation for layer {layer_idx}, head {head_idx}")
        print(f"DEBUG: *** NO FILE SAVING MODE ***")
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
        
        print(f"Head ablation completed! (No summary file saved - debug mode)")
        return {
            "baseline_accuracy": baseline_accuracy,
            "head_result": {
                "layer": layer_idx, 
                "head": head_idx, 
                "accuracy": accuracy, 
                "delta": delta
            }
        }
    
    def run_layer_range_ablation(self, start_layer=0, end_layer=5):
        """Run layer range ablation experiment - NO FILE SAVING"""
        print(f"DEBUG: Starting layer range ablation {start_layer}-{end_layer}")
        print(f"DEBUG: *** NO FILE SAVING MODE ***")
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
        
        print(f"Range ablation completed! (No summary file saved - debug mode)")
        return {
            "baseline_accuracy": baseline_accuracy,
            "range_result": {
                "start": start_layer, 
                "end": end_layer, 
                "accuracy": accuracy, 
                "delta": delta
            }
        }
    
    def run_permutation(self, start_layer=0, end_layer=5):
        """Run layer permutation experiment - NO FILE SAVING"""
        print(f"DEBUG: Starting layer permutation {start_layer}-{end_layer}")
        print(f"DEBUG: *** NO FILE SAVING MODE ***")
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
        
        print(f"Permutation completed! (No summary file saved - debug mode)")
        return {
            "baseline_accuracy": baseline_accuracy,
            "permute_result": {
                "start": start_layer, 
                "end": end_layer, 
                "accuracy": accuracy, 
                "delta": delta, 
                "permutation": permutation_result
            }
        }