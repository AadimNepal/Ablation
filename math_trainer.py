from base_trainer import BaseTrainer
from math_verify import parse


class MathTrainer(BaseTrainer):
    """Trainer specialized for math problems (GSM8K)"""
    
    def __init__(self, model, dataset, num_problems=50, batch_size=16, model_name="unknown", dataset_name="math"):
        super().__init__(model, dataset, num_problems, batch_size, model_name, dataset_name)
    
    def extract_answer(self, output):
        """Extract numerical answer from math model output"""
        result = parse(output)
        if isinstance(result, list) and len(result) > 0:
            if len(result) > 1 and isinstance(result[1], str):
                return result[1]
            return str(result[0])
        return None
    
    def prepare_prompt(self, item):
        """Prepare prompt for math problem"""
        question = item['question']
        
        # Use special prompts only for distilled models
        if "distilled" in self.model_name.lower():
            if "deepseek" in self.model_name.lower():
                return (
                    "Solve this math problem step by step. Be concise but complete. "
                    "After solving, write your FINAL ANSWER as '\\boxed{your_answer}' on a new line.\n\n"
                    f"Question: {question}\n"
                    "Solution:"
                )
            elif "llama" in self.model_name.lower():
                return (
                    "<｜User｜>Solve the following math problem step by step. "
                    "Show your reasoning clearly and provide the final answer as '\\boxed{your_answer}'.\n\n"
                    f"Problem: {question}\n"
                    "<｜Assistant｜>"
                )
        
        # Default: just return the question for all other models
        return question
    
    def get_ground_truth(self, item):
        """Get ground truth answer from GSM8K item"""
        return item['final_answer']
    
    def check_correctness(self, prediction, ground_truth):
        """Check if prediction matches ground truth for math problems"""
        if prediction is None:
            return False
        return prediction.strip() == ground_truth.strip()