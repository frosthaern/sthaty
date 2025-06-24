from modules.classes.AttentionExtractor import AttentionExtractor
from typing import Generator
import numpy as np

class AttentionAnalyzer:
  def __init__(self, extractor: AttentionExtractor):
    self.extractor = extractor

  def print_info(self):
    for a in self.extractor:
      print(a["heads"].shape)
      print(a["special_ids"])

  def entropy(self) -> Generator[np.ndarray[np.float64, np.dtype[np.float64]], None, None]:
    n_layers, n_heads = 12, 12
    for attn in self.extractor:
      qb, qe, cb, ce, = attn["special_ids"]
      entropy = np.zeros((n_layers, n_heads), dtype=np.float64)
      for layer in range(n_layers):
        for head in range(n_heads):
          head_attention_score = attn["heads"][layer][head][qb+1:qe, cb+1:ce].numpy()
          row_sum = head_attention_score.sum(axis=1, keepdims=True)
          normalized_attention = head_attention_score / (row_sum + 1e-10)
          token_entropies = -np.sum(normalized_attention * np.log(normalized_attention + 1e-10), axis=1)
          entropy[layer][head] = np.mean(token_entropies)
      yield entropy # this is a 12 x 12 np array

  # have to try umap
  # then pca -> umap pipeline
  # then i have to see which one is giving better results
