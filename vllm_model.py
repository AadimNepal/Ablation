from vllm import LLM, SamplingParams
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import os, gc, copy
import random


class BaseVllmModel:
    """Base class for all vLLM models with common functionality"""
    
    def __init__(self, model_id, checkpoint_path, **llm_kwargs):
        self.model_id = model_id
        self.checkpoint_path = checkpoint_path
        
        print("Loading HuggingFace model...")
        self.hf_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Default LLM parameters
        default_params = {
            "trust_remote_code": True,
            "dtype": "half",
            "max_seq_len_to_capture": 300,
            "gpu_memory_utilization": 0.9
        }
        default_params.update(llm_kwargs)
        
        self.llm = LLM(model=model_id, **default_params)
        print("HuggingFace model loaded.")
        
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1000)

    def generate(self, prompt, **kwargs):
        if isinstance(prompt, str):
            params = self.sampling_params
            if kwargs:
                params = SamplingParams(**{**vars(self.sampling_params), **kwargs})
            
            outputs = self.llm.generate(prompt, params)
            return outputs[0].outputs[0].text
        else:
            raise ValueError("For batch processing, use generate_batch instead")
    
    def generate_batch(self, prompts, **kwargs):
        params = self.sampling_params
        if kwargs:
            params = SamplingParams(**{**vars(self.sampling_params), **kwargs})
        
        batch_outputs = self.llm.generate(prompts, params)
        results = [output.outputs[0].text for output in batch_outputs]
        return results

    def _cleanup_and_copy(self, layer_number):
        """Clean up vLLM and create model copy"""
        del self.llm
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Creating deep copy for layer {layer_number} ablation...")
        return copy.deepcopy(self.hf_model)

    def _save_and_reload(self, model_copy, **llm_kwargs):
        """Save ablated model and reload in vLLM"""
        # Generate random 5-digit suffix to avoid conflicts
        random_suffix = random.randint(10000, 99999)
        unique_checkpoint_path = f"{self.checkpoint_path}_{random_suffix}"
        
        print(f"Saving ablated model to {unique_checkpoint_path}...")
        os.makedirs(unique_checkpoint_path, exist_ok=True)
        
        # Clean up existing files
        if os.path.exists(unique_checkpoint_path):
            for file in os.listdir(unique_checkpoint_path):
                file_path = os.path.join(unique_checkpoint_path, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        
        model_copy.save_pretrained(unique_checkpoint_path)
        self.tokenizer.save_pretrained(unique_checkpoint_path)
        
        # Clean up
        del model_copy
        torch.cuda.empty_cache()
        gc.collect()
        
        print("Loading ablated model in VLLM...")
        # Default LLM parameters for reloading
        default_params = {
            "trust_remote_code": True,
            "dtype": "half",
            "max_seq_len_to_capture": 300,
            "gpu_memory_utilization": 0.9
        }
        default_params.update(llm_kwargs)
        
        self.llm = LLM(model=unique_checkpoint_path, **default_params)

    def zero_ablate(self, layer_number):
        """Zero ablate a single layer - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement zero_ablate method")


class QwenModelMixin:
    """Mixin for Qwen-based models"""
    
    def _ablate_layer(self, layer, layer_number):
        """Ablate a Qwen-style layer"""
        try:
            # Zero out self-attention weights
            attn = layer.self_attn
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                proj = getattr(attn, proj_name)
                proj.weight.data.zero_()
                if hasattr(proj, 'bias') and proj.bias is not None:
                    proj.bias.data.zero_()
            
            # Zero out MLP weights
            mlp = layer.mlp
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                if hasattr(mlp, proj_name):
                    proj = getattr(mlp, proj_name)
                    proj.weight.data.zero_()
                    if hasattr(proj, 'bias') and proj.bias is not None:
                        proj.bias.data.zero_()
        
        except AttributeError as e:
            print(f"Error accessing model architecture: {e}")
            raise


class LlamaModelMixin:
    """Mixin for Llama-based models"""
    
    def _ablate_layer(self, layer, layer_number):
        """Ablate a Llama-style layer"""
        try:
            # Zero out self-attention weights
            attn = layer.self_attn
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                proj = getattr(attn, proj_name)
                proj.weight.data.zero_()
                if hasattr(proj, 'bias') and proj.bias is not None:
                    proj.bias.data.zero_()
            
            # Zero out MLP weights
            mlp = layer.mlp
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                proj = getattr(mlp, proj_name)
                proj.weight.data.zero_()
                if hasattr(proj, 'bias') and proj.bias is not None:
                    proj.bias.data.zero_()
        
        except AttributeError as e:
            print(f"Error accessing model architecture: {e}")
            raise


class Qwen7BChat(BaseVllmModel, QwenModelMixin):
    def __init__(self):
        super().__init__(
            model_id="Qwen/Qwen-7B-Chat",
            checkpoint_path="./ablated_model_qwen7bchat"
        )
    
    def _format_prompt(self, prompt):
        """Format prompt for chat model compatibility"""
        return prompt

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...")
        layer = model_copy.transformer.h[layer_number]  # Qwen chat uses transformer.h
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(model_copy)
        return f"Layer {layer_number} ablated successfully"


class Qwen257BBase(BaseVllmModel, QwenModelMixin):
    def __init__(self):
        super().__init__(
            model_id="Qwen/Qwen2.5-7B",
            checkpoint_path="./ablated_model_qwen257bbase",
            dtype="half",
            max_model_len=32768,  
            max_seq_len_to_capture=1000,
            gpu_memory_utilization=0.7
        )

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...")
        layer = model_copy.model.layers[layer_number]
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(model_copy, max_seq_len_to_capture=1000)
        return f"Layer {layer_number} ablated successfully"


class Qwen257BInstruct(BaseVllmModel, QwenModelMixin):
    def __init__(self):
        super().__init__(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            checkpoint_path="./ablated_model_qwen257binstruct"
        )
        # Override sampling params for instruct model
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...")
        layer = model_copy.model.layers[layer_number]
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(model_copy)
        return f"Layer {layer_number} ablated successfully"


class DeepSeekR1DistillQwen7B(BaseVllmModel, QwenModelMixin):
    def __init__(self):
        super().__init__(
            model_id="deepseek-ai/deepseek-R1-Distill-Qwen-7B",
            checkpoint_path="./ablated_model_deepseekqwen7b",
            dtype="float16",
            max_model_len=1000,
            gpu_memory_utilization=0.85
        )
        # Override sampling params
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=10)

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...")
        layer = model_copy.model.layers[layer_number]
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(
            model_copy,
            dtype="float16",
            max_model_len=1000,
            gpu_memory_utilization=0.85
        )
        return f"Layer {layer_number} ablated successfully"


class Llama318BBase(BaseVllmModel, LlamaModelMixin):
    def __init__(self):
        super().__init__(
            model_id="meta-llama/Llama-3.1-8B",
            checkpoint_path="./ablated_model_llama318bbasedkkk",
            max_model_len=31768,
            gpu_memory_utilization=0.9
        )
        # Override sampling params
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1000)

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...")
        layer = model_copy.model.layers[layer_number]
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(
            model_copy,
            max_model_len=31768,
            gpu_memory_utilization=0.9
        )
        return f"Layer {layer_number} ablated successfully"


class Llama318BInstruct(BaseVllmModel, LlamaModelMixin):
    def __init__(self):
        super().__init__(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            checkpoint_path="./ablated_model_llama318binstruct",
            max_model_len=31768
        )
        # Override sampling params
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...")
        layer = model_copy.model.layers[layer_number]
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(model_copy, max_model_len=31768)
        return f"Layer {layer_number} ablated successfully"


class DeepSeekR1DistilledLlama(BaseVllmModel, LlamaModelMixin):
    def __init__(self):
        super().__init__(
            model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            checkpoint_path="./ablated_model_deepseek_r1_8b",
            max_model_len=32768,
            gpu_memory_utilization=0.90
        )

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...")
        layer = model_copy.model.layers[layer_number]
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(
            model_copy,
            max_model_len=32768,
            gpu_memory_utilization=0.90
        )
        return f"Layer {layer_number} ablated successfully"


class OpenReasonerBase(BaseVllmModel, QwenModelMixin):
    def __init__(self):
        super().__init__(
            model_id="Open-Reasoner-Zero/Open-Reasoner-Zero-7B",
            checkpoint_path="./ablated_model_open_reasoner_zero",
            max_model_len=32768
        )

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...")
        layer = model_copy.model.layers[layer_number]
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(model_copy, max_model_len=32768)
        return f"Layer {layer_number} ablated successfully"


class Llama31SimpleRLZoo(BaseVllmModel, LlamaModelMixin):
    def __init__(self):
        super().__init__(
            model_id="hkust-nlp/Llama-3.1-8B-SimpleRL-Zoo",
            checkpoint_path="./ablated_model_llama318bsimplerl",
            max_model_len=31768,
            gpu_memory_utilization=0.9
        )
        # Override sampling params
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...")
        layer = model_copy.model.layers[layer_number]
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(
            model_copy,
            max_model_len=31768,
            gpu_memory_utilization=0.9
        )
        return f"Layer {layer_number} ablated successfully"