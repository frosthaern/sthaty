from typing import Generator, Dict
from torch import Tensor


class AttentionAnalyzer:
  def __init__(self, encodings: Generator[Dict[str, Tensor], None, None]):
    self.encodings = encodings

  def analyze(self):
    pass
