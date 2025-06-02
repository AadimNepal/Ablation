from datasets import load_dataset
import re

class GSM8KLoader:
    def __init__(self, split="train"):
        self.dataset = load_dataset("gsm8k", "main")[split]
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item["question"]
        answer_text = item["answer"]
        solution_steps = answer_text.split("####")[0].strip()
        answer_match = re.search(r"####\s*(\d+)", answer_text)
        
        if answer_match:
            final_answer = answer_match.group(1)
        else:
            print(f"Warning: Example {idx} has missing answer format. Replacing with dummy problem.")
            question = "What is 3333 times 22222?"
            solution_steps = "3333 times 22222 would be 74106426."
            final_answer = "74106426"
            
        return {
            "question": question,
            "solution_steps": solution_steps,
            "final_answer": final_answer
        }
        
    def __len__(self):
        return len(self.dataset)


class TriviaQALoader:
    def __init__(self, split="train", config="rc.nocontext"):
        """
        Load TriviaQA dataset
        Args:
            split: "train", "validation", or "test" 
            config: "rc.nocontext" (clean Q&A) or "unfiltered.nocontext"
        """
        self.dataset = load_dataset("trivia_qa", config)[split]
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        question = item["question"]
        answer_info = item["answer"]
        
        # Get the primary answer value
        primary_answer = answer_info["value"]
        
        # Get all possible answer aliases (for flexible matching)
        answer_aliases = answer_info.get("aliases", [])
        if primary_answer not in answer_aliases:
            answer_aliases = [primary_answer] + answer_aliases
            
        # Normalize aliases for easier matching
        normalized_aliases = [alias.lower().strip() for alias in answer_aliases]
        
        return {
            "question": question,
            "answer": primary_answer,
            "answer_aliases": answer_aliases,
            "normalized_aliases": normalized_aliases,
            "question_id": item.get("question_id", ""),
            "question_source": item.get("question_source", "")
        }
        
    def __len__(self):
        return len(self.dataset)

class MMLULoader:
    def __init__(self, split="test", subject_filter=None, question_type_filter=None):
        """
        Load MMLU dataset with filtering options
        
        Args:
            split: "test", "validation", or "dev" 
            subject_filter: List of specific subjects to include, or None for all
            question_type_filter: "factual", "reasoning", or None for all
        """
        # Load the full MMLU dataset
        self.dataset = load_dataset("cais/mmlu", "all")[split]
        
        # Define factual vs reasoning subject categorization
        # Based on literature analysis - factual subjects focus on knowledge recall
        self.factual_subjects = {
            'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 
            'clinical_knowledge', 'college_biology', 'college_chemistry', 
            'college_computer_science', 'college_mathematics', 'college_medicine', 
            'college_physics', 'computer_security', 'conceptual_physics', 
            'econometrics', 'electrical_engineering', 'elementary_mathematics', 
            'formal_logic', 'global_facts', 'high_school_biology', 
            'high_school_chemistry', 'high_school_computer_science', 
            'high_school_european_history', 'high_school_geography', 
            'high_school_government_and_politics', 'high_school_macroeconomics', 
            'high_school_mathematics', 'high_school_microeconomics', 
            'high_school_physics', 'high_school_psychology', 'high_school_statistics', 
            'high_school_us_history', 'high_school_world_history', 
            'human_anatomy', 'human_sexuality', 'international_law', 
            'jurisprudence', 'logical_fallacies', 'machine_learning', 
            'management', 'marketing', 'medical_genetics', 'miscellaneous', 
            'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 
            'professional_law', 'professional_medicine', 'professional_psychology', 
            'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 
            'virology', 'world_religions'
        }
        
        # Reasoning subjects focus more on logical deduction and problem-solving
        self.reasoning_subjects = {
            'logical_fallacies', 'moral_scenarios', 'philosophy', 
            'formal_logic', 'abstract_algebra', 'college_mathematics', 
            'elementary_mathematics', 'high_school_mathematics', 
            'high_school_statistics', 'econometrics', 'machine_learning',
            'college_computer_science', 'high_school_computer_science',
            'computer_security', 'electrical_engineering', 'college_physics',
            'conceptual_physics', 'high_school_physics', 'college_chemistry',
            'high_school_chemistry', 'professional_law', 'jurisprudence',
            'international_law', 'business_ethics'
        }
        
        # Apply filtering
        self.filtered_data = self._apply_filters(subject_filter, question_type_filter)
        
    def _apply_filters(self, subject_filter, question_type_filter):
        """Apply subject and question type filters to the dataset"""
        filtered_data = []
        
        for item in self.dataset:
            subject = item['subject']
            
            # Apply subject filter
            if subject_filter is not None:
                if subject not in subject_filter:
                    continue
            
            # Apply question type filter
            if question_type_filter == "factual":
                if subject not in self.factual_subjects:
                    continue
            elif question_type_filter == "reasoning":
                if subject not in self.reasoning_subjects:
                    continue
            # If question_type_filter is None, include all
            
            filtered_data.append(item)
        
        return filtered_data
    
    def __getitem__(self, idx):
        """Get item from filtered dataset"""
        item = self.filtered_data[idx]
        
        question = item['question']
        choices = item['choices']
        answer_idx = item['answer']  # This is the index (0-3) of the correct answer
        answer_letter = chr(ord('A') + answer_idx)  # Convert to letter (A, B, C, D)
        answer_text = choices[answer_idx]  # Get the actual answer text
        subject = item['subject']
        
        return {
            "question": question,
            "choices": choices,
            "answer_idx": answer_idx,
            "answer_letter": answer_letter,
            "answer_text": answer_text,
            "subject": subject,
            "question_type": self._get_question_type(subject)
        }
    
    def _get_question_type(self, subject):
        """Determine if a subject is primarily factual or reasoning-based"""
        if subject in self.reasoning_subjects:
            return "reasoning"
        else:
            return "factual"
    
    def __len__(self):
        return len(self.filtered_data)
    
    def get_subjects(self):
        """Get list of unique subjects in the filtered dataset"""
        return list(set(item['subject'] for item in self.filtered_data))
    
    def get_stats(self):
        """Get statistics about the filtered dataset"""
        subjects = {}
        question_types = {"factual": 0, "reasoning": 0}
        
        for item in self.filtered_data:
            subject = item['subject']
            subjects[subject] = subjects.get(subject, 0) + 1
            
            q_type = self._get_question_type(subject)
            question_types[q_type] += 1
        
        return {
            "total_questions": len(self.filtered_data),
            "subjects": subjects,
            "question_types": question_types,
            "num_subjects": len(subjects)
        }
