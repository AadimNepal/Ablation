from vllm import LLM, SamplingParams
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import os, gc, copy
import random
from datetime import datetime
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory


class BaseVllmModel:
    """Base class for all vLLM models with common functionality"""
    
    def __init__(self, model_id, checkpoint_path, **llm_kwargs):
        self.model_id = model_id
        self.checkpoint_path = checkpoint_path
        
        print("Loading HuggingFace model...")
        self.hf_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Default LLM parameters with enforce_eager=True
        default_params = {
            "trust_remote_code": True,
            "dtype": "half",
            "max_seq_len_to_capture": 300,
            "gpu_memory_utilization": 0.9,
            "enforce_eager": True  # ADDED: Disable compilation
        }
        default_params.update(llm_kwargs)
        
        self.llm = LLM(model=model_id, **default_params)
        print("HuggingFace model loaded.")
        
        # Consistent sampling params with max_tokens=1000
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
        """Enhanced cleanup and create model copy"""
        import time
        import subprocess
        
        print(f"Enhanced cleanup for layer {layer_number}...")
        
        # Kill any hanging vLLM processes
        try:
            subprocess.run(['pkill', '-f', 'vllm'], check=False, timeout=5)
            time.sleep(2)
        except:
            pass

        # Clean up vLLM distributed state
        try:
            cleanup_dist_env_and_memory()
            print("✓ vLLM distributed cleanup done")
        except Exception as e:
            print(f"Warning: vLLM cleanup failed: {e}")

        # Delete LLM instance
        if hasattr(self, 'llm') and self.llm is not None:
            try:
                del self.llm
                self.llm = None
                print("✓ LLM object deleted")
            except Exception as e:
                print(f"Warning: LLM deletion failed: {e}")
        
        # Aggressive CUDA cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            try:
                torch.cuda.ipc_collect()
            except:
                pass
            print("✓ CUDA cache cleared")
        
        # Multiple garbage collection passes
        for i in range(3):
            collected = gc.collect()
            time.sleep(0.5)
            print(f"✓ GC run {i+1}: collected {collected} objects")
        
        # Check memory status
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            print(f"✓ GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
        
        print(f"Creating deep copy for layer {layer_number} ablation...")
        return copy.deepcopy(self.hf_model)

    def _save_and_reload(self, model_copy, **llm_kwargs):
        """Save ablated model and reload in vLLM with consistent parameters"""
        print("Saving ablated model...")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        # Clean up existing files
        if os.path.exists(self.checkpoint_path):
            for file in os.listdir(self.checkpoint_path):
                file_path = os.path.join(self.checkpoint_path, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        
        model_copy.save_pretrained(self.checkpoint_path)
        self.tokenizer.save_pretrained(self.checkpoint_path)
        
        # Clean up
        del model_copy
        torch.cuda.empty_cache()
        gc.collect()
        
        print("Loading ablated model in VLLM...")
        # CONSISTENT parameters for reloading with enforce_eager=True
        reload_params = {
            "trust_remote_code": True,
            "dtype": "half",
            "max_seq_len_to_capture": 300,
            "gpu_memory_utilization": 0.9,
            "enforce_eager": True  # CRITICAL: Always disable compilation
        }
        reload_params.update(llm_kwargs)
        
        self.llm = LLM(model=self.checkpoint_path, **reload_params)

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
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        super().__init__(
            model_id="Qwen/Qwen-7B-Chat",
            checkpoint_path=f"./ablated_model_qwen7bchat_{timestamp}",
            enforce_eager=True
        )
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1000)
    
    def _format_prompt(self, prompt):
        """Format prompt for chat model compatibility"""
        return prompt

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...")
        layer = model_copy.transformer.h[layer_number]  # Qwen chat uses transformer.h
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(model_copy, enforce_eager=True)
        return f"Layer {layer_number} ablated successfully"


class Qwen257BBase(BaseVllmModel, QwenModelMixin):
    def __init__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        super().__init__(
            model_id="Qwen/Qwen2.5-7B",
            checkpoint_path=f"./ablated_model_qwen257bbase_{timestamp}",
            dtype="half",
            max_model_len=32768,  
            max_seq_len_to_capture=1000,
            gpu_memory_utilization=0.7,
            enforce_eager=True
        )
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1000)

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...")
        layer = model_copy.model.layers[layer_number]
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(
            model_copy, 
            max_seq_len_to_capture=1000,
            enforce_eager=True
        )
        return f"Layer {layer_number} ablated successfully"


class Qwen257BInstruct(BaseVllmModel, QwenModelMixin):
    def __init__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        super().__init__(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            checkpoint_path=f"./ablated_model_qwen257binstruct_{timestamp}",
            enforce_eager=True
        )
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1000)

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...")
        layer = model_copy.model.layers[layer_number]
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(model_copy, enforce_eager=True)
        return f"Layer {layer_number} ablated successfully"


class DeepSeekR1DistillQwen7B(BaseVllmModel, QwenModelMixin):
    def __init__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        super().__init__(
            model_id="deepseek-ai/deepseek-R1-Distill-Qwen-7B",
            checkpoint_path=f"./ablated_model_deepseekqwen7b_{timestamp}",
            dtype="float16",
            max_model_len=1000,
            gpu_memory_utilization=0.85,
            enforce_eager=True
        )
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1000)

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...")
        layer = model_copy.model.layers[layer_number]
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(
            model_copy,
            dtype="float16",
            max_model_len=1000,
            gpu_memory_utilization=0.85,
            enforce_eager=True
        )
        return f"Layer {layer_number} ablated successfully"


class Llama318BBase(BaseVllmModel, LlamaModelMixin):
    def __init__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        super().__init__(
            model_id="meta-llama/Llama-3.1-8B",
            checkpoint_path=f"./ablated_model_llama318bbase_{timestamp}",
            max_model_len=31768,
            gpu_memory_utilization=0.9,
            enforce_eager=True
        )
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1000)

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...")
        layer = model_copy.model.layers[layer_number]
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(
            model_copy,
            max_model_len=31768,
            gpu_memory_utilization=0.9,
            enforce_eager=True
        )
        return f"Layer {layer_number} ablated successfully"


class Llama318BInstruct(BaseVllmModel, LlamaModelMixin):
    def __init__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        super().__init__(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            checkpoint_path=f"./ablated_model_llama318binstruct_{timestamp}",
            max_model_len=31768,
            enforce_eager=True
        )
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1000)

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...")
        layer = model_copy.model.layers[layer_number]
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(
            model_copy, 
            max_model_len=31768,
            enforce_eager=True
        )
        return f"Layer {layer_number} ablated successfully"


class DeepSeekR1DistilledLlama(BaseVllmModel, LlamaModelMixin):
    def __init__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        super().__init__(
            model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            checkpoint_path=f"./ablated_model_deepseek_r1_8b_{timestamp}",
            max_model_len=32768,
            gpu_memory_utilization=0.90,
            enforce_eager=True
        )
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1000)

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...")
        layer = model_copy.model.layers[layer_number]
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(
            model_copy,
            max_model_len=32768,
            gpu_memory_utilization=0.90,
            enforce_eager=True
        )
        return f"Layer {layer_number} ablated successfully"


class OpenReasonerBase(BaseVllmModel, QwenModelMixin):
    def __init__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        super().__init__(
            model_id="Open-Reasoner-Zero/Open-Reasoner-Zero-7B",
            checkpoint_path=f"./ablated_model_open_reasoner_zero_{timestamp}",
            max_model_len=32768,
            enforce_eager=True
        )
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1000)

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...")
        layer = model_copy.model.layers[layer_number]
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(
            model_copy, 
            max_model_len=32768,
            enforce_eager=True
        )
        return f"Layer {layer_number} ablated successfully"


class Llama31SimpleRLZoo(BaseVllmModel, LlamaModelMixin):
    def __init__(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        super().__init__(
            model_id="hkust-nlp/Llama-3.1-8B-SimpleRL-Zoo",
            checkpoint_path=f"./ablated_model_llama318bsimplerl_{timestamp}",
            max_model_len=31768,
            gpu_memory_utilization=0.9,
            enforce_eager=True
        )
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1000)

    def zero_ablate(self, layer_number):
        model_copy = self._cleanup_and_copy(layer_number)
        
        print(f"Ablating layer {layer_number}...")
        layer = model_copy.model.layers[layer_number]
        self._ablate_layer(layer, layer_number)
        
        self._save_and_reload(
            model_copy,
            max_model_len=31768,
            gpu_memory_utilization=0.9,
            enforce_eager=True
        )
        return f"Layer {layer_number} ablated successfully"