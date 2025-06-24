from typing import Dict, Generator, List, Any
from torch import Tensor
from transformers import AutoTokenizer
from tqdm import tqdm

from modules.absclasses.LoadDataAbs import LoadDataAbs
from modules.absclasses.LoadTokenizedDataAbs import LoadTokenizedDataAbs


class CodeBertTokenizeEncode(LoadTokenizedDataAbs):
  """Handles tokenization and encoding of code and queries using CodeBERT."""

  def __init__(
    self,
    data: LoadDataAbs,
    padding: str = "max_length",
    max_length: int = 512,
    truncation: bool = True,
    batch_size: int = 32,
    show_progress: bool = True,
  ):
    self.data = data
    self.padding = padding
    self.max_length = max_length
    self.truncation = truncation
    self.batch_size = batch_size
    self.show_progress = show_progress
    self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

  def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Tensor]]:
    """Process a batch of data through the tokenizer."""
    try:
      # Prepare batch inputs
      queries = [item.get("query", "") for item in batch]
      codes = [item.get("code", "") for item in batch]

      # Tokenize batch
      encodings = self.tokenizer(
        queries,
        codes,
        return_tensors="pt",
        padding=self.padding,
        max_length=self.max_length,
        truncation=self.truncation,
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=True,
      )

      # Process each item in the batch
      results = []
      for i in range(len(batch)):
        item_encoding = {k: v[i] for k, v in encodings.items()}
        token_ids = item_encoding["input_ids"].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        special_ids = [i for i, t in enumerate(tokens) if t in ["<s>", "</s>"]]
        item_encoding["special_ids"] = special_ids
        results.append(item_encoding)

      return results

    except Exception as e:
      print(f"Error processing batch: {str(e)}")
      return []

  def __iter__(self) -> Generator[Dict[str, Tensor], None, None]:
    """Iterate over the dataset, yielding tokenized and encoded items."""
    batch = []

    # Wrap the data iterator with tqdm for progress tracking
    data_iter = tqdm(self.data, desc="Tokenizing") if self.show_progress else self.data

    for item in data_iter:
      batch.append(item)
      if len(batch) >= self.batch_size:
        for encoded in self._process_batch(batch):
          if encoded:  # Only yield if encoding was successful
            yield encoded
        batch = []

    # Process the last batch if it's not empty
    if batch:
      for encoded in self._process_batch(batch):
        if encoded:
          yield encoded
