import json
from typing import Generator

from modules.absclasses.LoadDataAbs import LoadDataAbs


class Jsonl(LoadDataAbs):
  def __init__(self, path, max_lines=10):
    self.path = path
    self.max_lines = max_lines

  def __iter__(self) -> Generator[str, None, None]:
    try:
      with open(self.path, "r") as f:
        for i, line in enumerate(f):
          if i >= self.max_lines:
            break
          d = json.loads(line)
          yield f"[CLS] {d['query']} [SEP] {d['code']} [SEP]"
    except Exception as e:
      print(f"{e} happened while trying to open {self.path}")
      raise e
