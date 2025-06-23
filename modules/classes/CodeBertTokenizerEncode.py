from typing import Dict, Generator

from torch import Tensor
from transformers import AutoTokenizer

from modules.absclasses.LoadDataAbs import LoadDataAbs
from modules.absclasses.LoadTokenizedDataAbs import LoadTokenizedDataAbs


class CodeBertTokenizeEncode(LoadTokenizedDataAbs):
  # this class will be used for tokenizing and encoding things here

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
    self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

  def __iter__(self) -> Generator[Dict[str, Tensor], None, None]:
    for data in self.data:
      try:
        encodings = self.tokenizer(
          data["query"],
          data["code"],
          return_tensors="pt",
          padding=self.padding,
          max_length=self.max_length,
          truncation=self.truncation,
          add_special_tokens=True,
        )
        token_ids = encodings["input_ids"][0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        special_ids = []
        for i, t in enumerate(tokens):
          if t in ["<s>", "</s>"]:
            special_ids.append(i)
        encodings["special_ids"] = special_ids
        yield encodings
      except Exception as e:
        print(f"{e} happened while tokenizeEncoding")
        raise e
