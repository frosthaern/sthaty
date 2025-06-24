from transformers import AutoModel
from typing import Generator, Dict
import torch
from collections import defaultdict
from torch import Tensor

from modules.absclasses.LoadTokenizedDataAbs import LoadTokenizedDataAbs


class AttentionExtractor:
  def __init__(self, encoder: LoadTokenizedDataAbs):
    self.encoder = encoder
    self.model = AutoModel.from_pretrained("microsoft/codebert-base", attn_implementation="eager")
    self.model.eval()

  def __iter__(self) -> Generator[Dict[str, Tensor], None, None]: 
    for encodings in self.encoder:
      attention_head_and_special_symbols = defaultdict(Tensor)
      with torch.no_grad():
        attention_heads =  self.model(
          input_ids=encodings["input_ids"],
          attention_mask=encodings["attention_mask"],
          output_attentions=True
        ).attentions
        heads_per_layer = []
        for ah in attention_heads:
          heads_per_layer.append(ah.squeeze())
        attention_head_and_special_symbols["heads"] = torch.stack(heads_per_layer, dim=0)
        attention_head_and_special_symbols["special_ids"] = encodings["special_ids"]
      yield attention_head_and_special_symbols
      
