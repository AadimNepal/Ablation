from base_trainer import BaseTrainer
from math_verify import parse


class MathTrainer(BaseTrainer):
    """Trainer specialized for math problems (GSM8K)"""
    
    def extract_answer(self, output):
        """Extract numerical answer from math model output"""
        result = parse(output)
        if isinstance(result, list) and len(result) > 0:
            if len(result) > 1 and isinstance(result[1], str):
                return result[1]
            return str(result[0])
        return None
    
    def prepare_prompt(self, item):
        """Prepare prompt for math problem - just the question"""
        return item['question']
    
    def get_ground_truth(self, item):
        """Get ground truth answer from GSM8K item"""
        return item['final_answer']
    
    def check_correctness(self, prediction, ground_truth):
        """Check if prediction matches ground truth for math problems"""
        if prediction is None:
            return False
        return prediction.strip() == ground_truth.strip()