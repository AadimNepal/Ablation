import re
from base_trainer import BaseTrainer

class TriviaTrainer(BaseTrainer):
    """Trainer specialized for TriviaQA factual recall"""
    
    def extract_answer(self, output):
        """Return the full output - we'll check if correct answer is contained within"""
        return output.strip() if output else ""
    
    def prepare_prompt(self, item):
        """Prepare prompt for trivia question - just the question"""
        return item['question']
    
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