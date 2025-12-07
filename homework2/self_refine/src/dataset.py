# dataset.py
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import ast 

# Dataset abstract class
class DatasetHandler(ABC):
    @abstractmethod
    def format_question(self, example: Dict[str, Any]) -> str: ...
    # TODO: format the question here based on the example structure

    @abstractmethod
    def parse_answer(self, response: str) -> Any: ...
    # TODO: parse the model response to extract the answer
    
    @abstractmethod
    def verify_answer(self, predicted: Any, ground_truth: Any) -> bool: ...
    # TODO: implement answer verification logic
    
    @abstractmethod
    def get_ground_truth(self, example: Dict[str, Any]) -> Any: ...
    # TODO: extract ground truth from the example


# You need to implement these handlers based on your datasets
# Each dataset can have its own parsing, verification logic

class GraphHandler(DatasetHandler):
    """Handler for graph pathfinding dataset."""

    def format_question(self, example: Dict[str, Any]) -> str:
        return example['prompt']

    def parse_answer(self, response: str) -> Any:
        """Parse model response to extract paths and weights."""
        import re

        # Remove thinking tokens 
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)

        paths = []
        weights = []

        # Look for patterns like "Path 1: [0, 1, 3, 4] (weight: 15)"
        pattern = r'Path\s+\d+:\s*\[([^\]]+)\]\s*\(weight:\s*(\d+)\)'
        matches = re.findall(pattern, response, re.IGNORECASE)

        for path_str, weight_str in matches:
            nodes = [int(n.strip()) for n in path_str.split(',')]
            paths.append(nodes)
            weights.append(int(weight_str))

        return {'paths': paths, 'weights': weights}

    def verify_answer(self, predicted: Any, ground_truth: Any) -> bool:
        """
        Verify if predicted answer matches ground truth.

        True if the model gets at least ONE of the predictions right.
        """
        if not isinstance(predicted, dict) or 'paths' not in predicted or 'weights' not in predicted:
            return False

        if not predicted['paths']:  # No predictions
            return False

        # Build list of predicted path-weight pairs
        pred_paths = [{"path": p, "weight": w} for p, w in zip(predicted['paths'], predicted['weights'])]

        # Build list of ground truth path-weight pairs
        gt_paths = [{"path": p, "weight": w} for p, w in zip(ground_truth['paths'], ground_truth['weights'])]

        # Check if at least one prediction matches any ground truth
        for pred_path_info in pred_paths:
            for gt_path_info in gt_paths:
                if (pred_path_info["path"] == gt_path_info["path"] and
                    pred_path_info["weight"] == gt_path_info["weight"]):
                    return True

        return False

    def get_ground_truth(self, example: Dict[str, Any]) -> Any:
        """
        Extract ground truth from example.

        Dataset format: {'solution': {'paths': [{'path': [...], 'weight': X}, ...]}}
        Convert to: {'paths': [[...], [...]], 'weights': [X, Y]}
        """
        solution = example['solution']

        paths = [item['path'] for item in solution['paths']]
        weights = [item['weight'] for item in solution['paths']]

        return {
            'paths': paths,
            'weights': weights
        }


class InfobenchHandler(DatasetHandler):
    """Handler for Infobench dataset."""

    def format_question(self, example: Dict[str, Any]) -> str:
        """Format Infobench question."""
        return example['question']

    def parse_answer(self, response: str) -> Any:
        """For Infobench, the response itself is the answer."""
        return response.strip()

    def verify_answer(self, predicted: Any, ground_truth: Any) -> bool:
        """Infobench requires LLM-based evaluation, not simple verification."""
        # This should use GPT-5-nano for evaluation
        # For now, return False as placeholder
        raise NotImplementedError("Infobench requires LLM-based evaluation")

    def get_ground_truth(self, example: Dict[str, Any]) -> Any:
        """Get reference answer if available."""
        return example.get('answer', '')


class MMLUMedHandler(DatasetHandler):
    """Handler for MMLU medical dataset."""

    def format_question(self, example: Dict[str, Any]) -> str:
        """Format MMLU multiple choice question."""
        question = example['question']
        choices = example['choices']

        prompt = f"{question}\n\n"
        for i, choice in enumerate(choices):
            letter = chr(65 + i)  # A, B, C, D
            prompt += f"{letter}. {choice}\n"

        prompt += "\nAnswer with just the letter (A, B, C, or D):"
        return prompt

    def parse_answer(self, response: str) -> Any:
        """Parse the answer letter from response."""
        import re

        # Remove thinking tokens
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)

        # Look for single letter A, B, C, or D 
        response_upper = response.upper()

        # "Final Answer: X" 
        match = re.search(r'FINAL\s+ANSWER\s*:\s*([A-D])\b', response_upper)
        if match:
            return match.group(1)

        # "The answer is X" or similar clear statements
        match = re.search(r'(?:ANSWER\s+IS|CHOOSE|SELECT)\s*:?\s*([A-D])\b', response_upper)
        if match:
            return match.group(1)

        return None

    def verify_answer(self, predicted: Any, ground_truth: Any) -> bool:
        """Check if predicted letter matches ground truth."""
        if predicted is None:
            return False

        # Ground truth is stored as integer 0-3 in the dataset
        # Convert to letter A-D for comparison
        if isinstance(ground_truth, int):
            gt_letter = chr(65 + ground_truth)  # 0->A, 1->B, 2->C, 3->D
        else:
            gt_letter = str(ground_truth).upper()

        pred_letter = str(predicted).upper()
        return pred_letter == gt_letter

    def get_ground_truth(self, example: Dict[str, Any]) -> Any:
        """Extract correct answer (returns integer 0-3 from dataset)."""
        return example['answer']

