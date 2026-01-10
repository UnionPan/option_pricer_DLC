from abc import ABC, abstractmethod
import numpy as np

class Derivative(ABC):
    """Base class for all derivative contracts"""

    def __init__(self, maturity: float):
        self.maturity = float(maturity)

    @abstractmethod
    def payoff(self, *args, **kwargs) -> np.ndarray:
        """
          Compute payoff at maturity
          
          Returns:
              Array of payoffs, shape (n_paths,)
        """
        pass

    @property
    @abstractmethod
    def contract_type(self) -> str:
        """Type of derivative (call, put, swap, etc.)"""
        pass


class PathIndependentDerivative(Derivative):
    """Derivatives that only depend on final value"""

    @abstractmethod
    def payoff(self, S_T: np.ndarray) -> np.ndarray:
        """Payoff depends only on terminal value S_T"""
        pass

class PathDependentDerivative(Derivative):
    """Derivatives that depend on entire path"""

    @abstractmethod
    def payoff(self, path: np.ndarray) -> np.ndarray:
        """
          Payoff depends on full path

          Args:
              path: Shape (n_steps, n_paths, dim)
        """
        pass

