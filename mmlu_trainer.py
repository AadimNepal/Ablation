import re
from base_trainer import BaseTrainer

class MMLUTrainer(BaseTrainer):
    """Trainer specialized for MMLU multiple choice questions"""
    
    def __init__(self, model, dataset, num_problems=50, batch_size=16, model_name="unknown", dataset_name="mmlu"):
        super().__init__(model, dataset, num_problems, batch_size, model_name, dataset_name)

    def extract_answer(self, output):
        """Extract the letter answer (A, B, C, D) from model output"""
        if not output:
            return None
            
        output = output.strip()
        
        # Try multiple patterns to extract the answer
        patterns = [
            r'(?:answer|Answer|ANSWER)(?:\s*is\s*|\s*:\s*|\s+)([A-D])',  # "Answer is A" or "Answer: A"
            r'(?:^|\s)([A-D])(?:\s*[\.\)]|\s|$)',  # Standalone letter with punctuation or space
            r'\(([A-D])\)',  # Letter in parentheses
            r'(?:option|Option|OPTION)(?:\s+)([A-D])',  # "Option A"
            r'(?:choice|Choice|CHOICE)(?:\s+)([A-D])',  # "Choice A"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                return match.group(1).upper()
        
        # If no clear pattern found, look for any letter A-D in the output
        letters = re.findall(r'[A-D]', output.upper())
        if letters:
            return letters[0]  # Return first found letter
            
        return None
    
    def prepare_prompt(self, item):
        """Prepare prompt for MMLU multiple choice question"""
        question = item['question']
        choices = item['choices']
        subject = item['subject']
        
        # Format choices as A, B, C, D
        formatted_choices = "\n".join([
            f"{chr(ord('A') + i)}. {choice}" 
            for i, choice in enumerate(choices)
        ])
        
        # Use special prompts for distilled models
        if "distilled" in self.model_name.lower():
            if "deepseek" in self.model_name.lower():
                return (
                    f"Subject: {subject.replace('_', ' ').title()}\n\n"
                    "Answer this multiple choice question by selecting the best option. "
                    "Provide your reasoning briefly, then clearly state your answer as 'Answer: X' where X is the letter.\n\n"
                    f"Question: {question}\n\n"
                    f"{formatted_choices}\n\n"
                    "Answer:"
                )
            elif "llama" in self.model_name.lower():
                return (
                    f"<｜User｜>Subject: {subject.replace('_', ' ').title()}\n\n"
                    "Please answer the following multiple choice question. "
                    "Think through the problem and then clearly indicate your answer as 'Answer: X' where X is the letter.\n\n"
                    f"Question: {question}\n\n"
                    f"{formatted_choices}\n"
                    "<｜Assistant｜>"
                )
        
        # Default prompt for other models
        return (
            f"The following is a multiple choice question about {subject.replace('_', ' ')}.\n\n"
            f"Question: {question}\n"
            f"{formatted_choices}\n"
            "Answer:"
        )
    
    def get_ground_truth(self, item):
        """Get ground truth answer letter from MMLU item"""
        return item['answer_letter']
    
    def check_correctness(self, prediction, ground_truth):
        """Check if prediction matches ground truth for MMLU"""
        if prediction is None:
            return False
        return prediction.strip().upper() == ground_truth.strip().upper()