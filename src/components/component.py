from abc import ABC, abstractmethod
import pandas as pd
from typing import Union

class Component(ABC):

    """Abstract base class for components."""
    @abstractmethod
    def load_data(self) -> Union[pd.DataFrame, tuple]:
        pass
