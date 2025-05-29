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