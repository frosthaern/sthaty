from transformers import AutoModel
import torch

from modules.absclasses.LoadTokenizedDataAbs import LoadTokenizedDataAbs


class AttentionExtractor:
  def __init__(self, encoder: LoadTokenizedDataAbs):
    self.encoder = encoder
    self.model = AutoModel.from_pretrained("microsoft/codebert-base", attn_implementation="eager")

  def __iter__(self): # add the shape of the tensor here
    for encodings in self.encoder:
      attention_heads =  self.model(**encodings, output_attentions=True).attentions
      heads = []
      for ah in attention_heads:
        heads.append(ah.squeeze())
      yield torch.stack(heads, dim=0)
