# code/pipeline/Evaluators/evals_utils/Evaluator.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class Evaluator(ABC):
    """
    Abstract base class for all evaluators.
    
    All evaluators must implement the evaluate() method which takes a model
    and returns a dictionary of metrics.
    """
    
    def __init__(self):
        pass
    
    @abstractmethod
    def evaluate(self, model: Any, **kwargs) -> Dict[str, float]:
        """
        Evaluate a model and return metrics.
        
        Args:
            model: The model to evaluate. Must have required interface methods.
            **kwargs: Additional arguments specific to the evaluator.
            
        Returns:
            Dictionary mapping metric names to their values.
        """
        pass
    
    @abstractmethod
    def get_dataset(self) -> Any:
        """
        Load and return the evaluation dataset.
        
        Returns:
            The dataset used for evaluation.
        """
        pass

    def get_dataloader(self, batch_size=1) -> Any:
        """
        Create and return a DataLoader for the evaluation dataset.
        
        Args:
            batch_size: Batch size for evaluation.
            
        Returns:
            DataLoader object.
        """
        pass