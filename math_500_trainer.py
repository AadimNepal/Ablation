from base_trainer import BaseTrainer
from math_verify import parse, verify

class Math500Trainer(BaseTrainer):
    """Trainer specialized for Math500 problems"""
    
    def __init__(self, model, dataset, num_problems=50, batch_size=16, model_name="unknown", dataset_name="math500", dataset_config=""):
        super().__init__(model, dataset, num_problems, batch_size, model_name, dataset_name, dataset_config)
    
    def extract_answer(self, output):
        """Extract answer from Math500 model output using math_verify"""
        result = parse(output)
        if isinstance(result, list) and len(result) > 0:
            if len(result) > 1 and isinstance(result[1], str):
                return result[1]
            return str(result[0])
        return None
    
    def prepare_prompt(self, item):
        """Prepare prompt for Math500 problem"""
        problem = item['problem']
        category = item['category']
        
        # Use special prompts for distilled models
        if "distilled" in self.model_name.lower():
            if "deepseek" in self.model_name.lower():
                return (
                    f"Solve this {category} problem step by step. Show your work clearly "
                    "and provide the final answer in \\boxed{} format.\n\n"
                    f"Problem: {problem}\n"
                    "Solution:"
                )
            elif "llama" in self.model_name.lower():
                return (
                    f"<｜User｜>Solve the following {category} problem step by step. "
                    "Show your reasoning and provide the final answer in \\boxed{} format.\n\n"
                    f"Problem: {problem}\n"
                    "<｜Assistant｜>"
                )
        
        # Default prompt for other models
        return problem
    
    def get_ground_truth(self, item):
        """Get ground truth answer from Math500 item"""
        return item['final_answer']
    
    def check_correctness(self, prediction, ground_truth):
        """Check if prediction matches ground truth for math problems using math_verify"""
        if prediction is None:
            return False
        
        # Parse both ground truth and prediction
        gold = parse(ground_truth)
        answer = parse(prediction)
        
        # Order is important: verify(gold, answer)
        return verify(gold, answer)
