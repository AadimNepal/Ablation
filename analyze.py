import torch
import matplotlib.pyplot as plt
import numpy as np
from vllm_model import Qwen257BBase
from gsm8 import GSM8KLoader

def collect_activations(model, problems, target_layers=range(20, 27)):
    """Collect max activations across specified layers"""
    activations_data = {layer: [] for layer in target_layers}
    
    for problem in problems:
        # Tokenize
        inputs = model.tokenizer(problem, return_tensors="pt", truncation=True, max_length=512)
        
        # Forward pass with hooks
        layer_outputs = {}
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activation = output[0].detach()
                else:
                    activation = output.detach()
                # Get max absolute activation for this layer
                layer_outputs[layer_idx] = torch.max(torch.abs(activation)).item()
            return hook
        
        # Register hooks
        hooks = []
        for layer_idx in target_layers:
            layer = model.hf_model.model.layers[layer_idx]
            hook = layer.register_forward_hook(make_hook(layer_idx))
            hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = model.hf_model(**inputs)
        
        # Store results
        for layer_idx in target_layers:
            activations_data[layer_idx].append(layer_outputs.get(layer_idx, 0))
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
    
    return activations_data

def plot_activation_analysis(math_data, non_math_data):
    """Create clean plots comparing math vs non-math activations"""
    layers = list(math_data.keys())
    
    # Calculate statistics
    math_means = [np.mean(math_data[layer]) for layer in layers]
    math_maxs = [np.max(math_data[layer]) for layer in layers]
    non_math_means = [np.mean(non_math_data[layer]) for layer in layers]
    non_math_maxs = [np.max(non_math_data[layer]) for layer in layers]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Mean activations
    ax1.plot(layers, math_means, 'o-', color='red', linewidth=2, markersize=8, label='Math Problems', alpha=0.8)
    ax1.plot(layers, non_math_means, 'o-', color='blue', linewidth=2, markersize=8, label='Non-Math Text', alpha=0.8)
    ax1.axvline(x=23, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Layer 23')
    ax1.set_xlabel('Layer Number', fontsize=12)
    ax1.set_ylabel('Mean Max Activation', fontsize=12)
    ax1.set_title('Mean Maximum Activations Across Layers', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(layers)
    
    # Plot 2: Max activations
    ax2.plot(layers, math_maxs, 's-', color='red', linewidth=2, markersize=8, label='Math Problems', alpha=0.8)
    ax2.plot(layers, non_math_maxs, 's-', color='blue', linewidth=2, markersize=8, label='Non-Math Text', alpha=0.8)
    ax2.axvline(x=23, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Layer 23')
    ax2.set_xlabel('Layer Number', fontsize=12)
    ax2.set_ylabel('Peak Max Activation', fontsize=12)
    ax2.set_title('Peak Maximum Activations Across Layers', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(layers)
    
    plt.tight_layout()
    plt.savefig('layer_activation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print some statistics
    print(f"\n=== LAYER 23 ANALYSIS ===")
    print(f"Math problems - Mean: {math_means[layers.index(23)]:.3f}, Max: {math_maxs[layers.index(23)]:.3f}")
    print(f"Non-math text - Mean: {non_math_means[layers.index(23)]:.3f}, Max: {non_math_maxs[layers.index(23)]:.3f}")
    print(f"Ratio (Math/Non-math) - Mean: {math_means[layers.index(23)]/non_math_means[layers.index(23)]:.2f}x")

def main():
    print("Loading model...")
    model = Qwen257BBase()
    
    print("Loading math problems...")
    dataset = GSM8KLoader()
    
    # Get 10 math problems
    math_problems = []
    for i in range(10):
        item = dataset[i]
        math_problems.append(item['question'])
    
    # Create some non-math control sentences
    non_math_text = [
        "The cat sat on the comfortable mat in the living room.",
        "Yesterday I went to the store to buy groceries for dinner.",
        "She loves reading books about history and ancient civilizations.",
        "The weather today is sunny with a gentle breeze blowing.",
        "Music has the power to inspire and heal people's hearts.",
        "Traveling to new places opens your mind to different cultures.",
        "Technology continues to evolve at an unprecedented pace today.",
        "Cooking delicious meals brings families together around the table.",
        "Exercise and healthy eating are important for maintaining wellness.",
        "Art museums showcase creativity and human expression throughout history."
    ]
    
    print("Analyzing math problems...")
    math_activations = collect_activations(model, math_problems)
    
    print("Analyzing non-math text...")
    non_math_activations = collect_activations(model, non_math_text)
    
    print("Creating plots...")
    plot_activation_analysis(math_activations, non_math_activations)
    
    print("Analysis complete! Check 'layer_activation_analysis.png'")

if __name__ == "__main__":
    main()