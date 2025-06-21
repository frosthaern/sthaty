from abc import ABC, abstractmethod
from typing import Generator, Dict

# any class that helps in loading the dataset should implement this, abstractmethod
# and it should return a generator of strings
# because the encoder class uses this to generate the dataset for encoding


class LoadDataAbs(ABC):
    @abstractmethod
    def __iter__(self) -> Generator[str, None, None]:
        raise NotImplementedError
