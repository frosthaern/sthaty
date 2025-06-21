from abc import ABC, abstractmethod
from typing import Dict, Generator

from torch import Tensor


class LoadTokenizedDataAbs(ABC):
  # i am just making sure that whatever,
  # inherits this will have a method that can convert the string gnerators into pytorch Tensor generators

  @abstractmethod
  def __iter__(self) -> Generator[Dict[str, Tensor], None, None]:
    raise NotImplementedError
