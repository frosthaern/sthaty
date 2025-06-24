import json
from typing import Generator, Dict, List
from pathlib import Path

from modules.absclasses.LoadDataAbs import LoadDataAbs


class Jsonl(LoadDataAbs):
  def __init__(self, path: str, max_lines: int = 10, batch_size: int = 32):
    self.path = Path(path)
    self.max_lines = max_lines
    self.batch_size = batch_size

  def _read_in_batches(self, file_handle, batch_size: int) -> Generator[List[str], None, None]:
    """Read lines in batches from file."""
    batch = []
    for i, line in enumerate(file_handle):
      if i >= self.max_lines:
        if batch:  # Yield any remaining items in the last batch
          yield batch
        break
      batch.append(line)
      if len(batch) >= batch_size:
        yield batch
        batch = []
    if batch:  # Don't forget the last batch
      yield batch

  def __iter__(self) -> Generator[Dict, None, None]:
    """Iterate over JSONL file, yielding one JSON object at a time."""
    try:
      with open(self.path, "r") as f:
        # Process file in batches for better I/O performance
        for batch in self._read_in_batches(f, self.batch_size):
          for line in batch:
            try:
              yield json.loads(line)
            except json.JSONDecodeError as e:
              print(f"Error decoding JSON on line: {line[:100]}... Error: {e}")
              continue
    except FileNotFoundError:
      raise FileNotFoundError(f"File not found: {self.path}")
    except Exception as e:
      print(f"Error processing {self.path}: {str(e)}")
      raise
