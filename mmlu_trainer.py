import re
from base_trainer import BaseTrainer

class MMLUTrainer(BaseTrainer):
    """Trainer specialized for MMLU multiple choice questions"""
    
    def __init__(self, model, dataset, num_problems=50, batch_size=16, model_name="unknown", dataset_name="mmlu", dataset_config=""):
        super().__init__(model, dataset, num_problems, batch_size, model_name, dataset_name, dataset_config)

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
    
    def check_correctness_with_subject(self, prediction, item):
        """Enhanced correctness check that also considers the subject"""
        is_correct = self.check_correctness(prediction, item['answer_letter'])
        return {
            'is_correct': is_correct,
            'subject': item['subject'],
            'question_type': item['question_type'],
            'predicted_letter': prediction,
            'correct_letter': item['answer_letter'],
            'predicted_text': item['choices'][ord(prediction) - ord('A')] if prediction and prediction in 'ABCD' else None,
            'correct_text': item['answer_text']
        }
    
    def evaluate_batch_with_subjects(self, layer="baseline"):
        """Enhanced evaluation that tracks performance by subject and question type"""
        correct, total = 0, 0
        results = []
        subject_performance = {}
        type_performance = {"factual": {"correct": 0, "total": 0}, 
                           "reasoning": {"correct": 0, "total": 0}}
        
        total_problems = min(self.num_problems, len(self.dataset))
        
        print(f"Evaluating {layer} - Processing {total_problems} problems...")
        
        # Process in batches
        for batch_idx in range(0, total_problems, self.batch_size):
            batch_end_idx = min(batch_idx + self.batch_size, total_problems)
            batch_items = [self.dataset[idx] for idx in range(batch_idx, batch_end_idx)]
            
            # Prepare prompts and ground truths
            batch_prompts = [self.prepare_prompt(item) for item in batch_items]
            batch_responses = self.model.generate_batch(batch_prompts)
            
            # Process each response
            for idx, (item, prompt, response) in enumerate(zip(batch_items, batch_prompts, batch_responses)):
                prediction = self.extract_answer(response)
                is_correct = self.check_correctness(prediction, item['answer_letter'])
                
                if is_correct:
                    correct += 1
                total += 1
                
                # Track subject performance
                subject = item['subject']
                if subject not in subject_performance:
                    subject_performance[subject] = {"correct": 0, "total": 0}
                subject_performance[subject]["total"] += 1
                if is_correct:
                    subject_performance[subject]["correct"] += 1
                
                # Track question type performance
                q_type = item['question_type']
                type_performance[q_type]["total"] += 1
                if is_correct:
                    type_performance[q_type]["correct"] += 1
                
                # Store detailed results
                result_entry = {
                    "index": batch_idx + idx,
                    "prompt": prompt,
                    "ground_truth": item['answer_letter'],
                    "ground_truth_text": item['answer_text'],
                    "model_response": response,
                    "extracted_prediction": prediction,
                    "is_correct": is_correct,
                    "subject": subject,
                    "question_type": q_type,
                    "all_choices": item['choices'],
                    "dataset_item": item
                }
                
                results.append(result_entry)
        
        # Calculate subject accuracies
        subject_accuracies = {
            subject: perf["correct"] / perf["total"] 
            for subject, perf in subject_performance.items()
        }
        
        # Calculate type accuracies
        type_accuracies = {
            q_type: perf["correct"] / perf["total"] if perf["total"] > 0 else 0
            for q_type, perf in type_performance.items()
        }
        
        overall_accuracy = correct / total
        
        print(f"Layer {layer} - Overall accuracy: {overall_accuracy:.4f} ({correct}/{total})")
        print(f"Factual accuracy: {type_accuracies['factual']:.4f}")
        print(f"Reasoning accuracy: {type_accuracies['reasoning']:.4f}")
        
        return {
            "overall_accuracy": overall_accuracy,
            "subject_accuracies": subject_accuracies,
            "type_accuracies": type_accuracies,
            "subject_performance": subject_performance,
            "type_performance": type_performance,
            "detailed_results": results
        }