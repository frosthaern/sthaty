import json
from typing import Generator

from .absclasses.LoadDataAbs import LoadDataAbs


class Jsonl(LoadDataAbs):
  def __init__(self, path):
    self.path = path

  def __iter__(self) -> Generator[str, None, None]:
    with open(self.path, "r") as f:
      for line in f:
        d = json.loads(line)
        yield f"[CLS] {d['query']} [SEP] {d['code']} [SEP]"
