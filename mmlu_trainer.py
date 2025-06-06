import re
from base_trainer import BaseTrainer
import os
from datetime import datetime

class MMLUTrainer(BaseTrainer):
    """Trainer specialized for MMLU multiple choice questions"""
    
    def __init__(self, model, dataset, num_problems=50, batch_size=16, model_name="unknown", dataset_name="mmlu"):
        super().__init__(model, dataset, num_problems, batch_size, model_name, dataset_name)
    
    def _create_experiment_directory(self):
        """Create MMLU-specific experiment directory with subject/question type info"""
        # Get dataset info to create descriptive name
        try:
            stats = self.dataset.get_stats()
            subjects = list(stats['subjects'].keys())
            question_types = stats['question_types']
            
            # Create a descriptive name based on filtering
            if len(subjects) > 5:
                subject_str = f"all-{len(subjects)}subjects"
            else:
                subject_str = "-".join(subjects[:3])
                if len(subjects) > 3:
                    subject_str += f"-plus{len(subjects)-3}more"
            
            # Add question type if filtered
            if question_types['factual'] == 0:
                type_str = "reasoning"
            elif question_types['reasoning'] == 0:
                type_str = "factual"
            else:
                type_str = "mixed"
                
        except:
            subject_str = "all"
            type_str = "mixed"
        
        # Create directory name: model-dataset-subjects-type-timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dir_name = f"{self.model_name}-{self.dataset_name}-{subject_str}-{type_str}-{timestamp}"
        
        experiment_path = os.path.join("results", dir_name)
        os.makedirs(experiment_path, exist_ok=True)
        
        return experiment_path
    
    def create_result_entry(self, item, prompt, response, ground_truth, prediction, is_correct, problem_index):
        """Create MMLU-specific result entry with choices, subject, and question type"""
        return {
            "problem_index": problem_index,
            "question": item['question'],
            "choices": item['choices'],
            "subject": item['subject'],
            "question_type": item['question_type'],
            "ground_truth": ground_truth,
            "ground_truth_text": item['answer_text'],
            "prompt": prompt,
            "model_response": response,
            "extracted_prediction": prediction,
            "is_correct": is_correct,
            "choice_details": {
                "answer_idx": item['answer_idx'],
                "answer_letter": item['answer_letter'],
                "predicted_letter": prediction,
                "all_choices": {chr(ord('A') + i): choice for i, choice in enumerate(item['choices'])}
            }
        }

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