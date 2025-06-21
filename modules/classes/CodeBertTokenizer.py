from typing import Dict, Generator

from torch import Tensor
from transformers import AutoTokenizer

from .absclasses.LoadDataAbs import LoadDataAbs
from .absclasses.LoadTokenizedDataAbs import LoadTokenizedDataAbs


class CodeBertTokenizeEncode(LoadTokenizedDataAbs):
  tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

  def __init__(
    self,
    data: LoadDataAbs,
    padding: str = "max_length",
    max_length: int = 512,
    truncation: bool = True,
  ):
    self.data: LoadDataAbs = data
    self.padding = padding
    self.max_length = max_length
    self.truncation = truncation

  def __iter__(self) -> Generator[Dict[str, Tensor], None, None]:
    for data in self.data:
      try:
        yield CodeBertTokenizeEncode.tokenizer(
          data,
          return_tensors="pt",
          padding=self.padding,
          max_length=self.max_length,
          truncation=self.truncation,
        )
      except Exception as e:
        print(f"{e} happened while tokenizeEncoding")
        raise e
