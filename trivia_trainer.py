import re
from base_trainer import BaseTrainer

class TriviaTrainer(BaseTrainer):
    """Trainer specialized for TriviaQA factual recall"""
    
    def __init__(self, model, dataset, num_problems=50, batch_size=16, model_name="unknown", dataset_name="trivia", dataset_config=""):
        super().__init__(model, dataset, num_problems, batch_size, model_name, dataset_name, dataset_config)
    
    def extract_answer(self, output):
        """Return the full output - we'll check if correct answer is contained within"""
        return output.strip() if output else ""
    
    def prepare_prompt(self, item):
        """Prepare prompt for trivia question"""
        question = item['question']
        
        # Use special prompts only for distilled models
        if "distilled" in self.model_name.lower():
            if "deepseek" in self.model_name.lower():
                return (
                    "Answer this trivia question directly and concisely. "
                    "Provide the answer clearly in your response.\n\n"
                    f"Question: {question}\n"
                    "Answer:"
                )
            elif "llama" in self.model_name.lower():
                return (
                    "<｜User｜>Answer the following trivia question directly and accurately. "
                    "Provide a clear, concise answer.\n\n"
                    f"Question: {question}\n"
                    "<｜Assistant｜>"
                )
        
        # Default: just return the question for all other models
        return question
    
    def get_ground_truth(self, item):
        """Get ground truth answer from TriviaQA item"""
        return item['answer']
    
    def check_correctness(self, prediction, ground_truth):
        """Check if ground truth answer appears anywhere in the model's response"""
        if not prediction:
            return False
            
        # Convert both to lowercase for case-insensitive matching
        pred_lower = prediction.lower()
        truth_lower = ground_truth.lower()
        
        # Check if the answer appears in the response
        return truth_lower in pred_lower
    
    def check_correctness_with_aliases(self, prediction, item):
        """Enhanced correctness check using all answer aliases"""
        if not prediction:
            return False
            
        pred_lower = prediction.lower()
        
        # Check against all normalized aliases
        for alias in item.get('normalized_aliases', []):
            if alias in pred_lower:
                return True
                
        return False