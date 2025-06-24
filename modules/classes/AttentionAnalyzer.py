from modules.classes.AttentionExtractor import AttentionExtractor
from sklearn.decomposition import PCA
from umap import UMAP
from typing import Generator, List
import numpy as np
from numpy.typing import NDArray

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

  def pca(self, vectors: List[NDArray[np.float64]]) -> Generator[NDArray[np.float64], None, None]:
    pca_reducer = PCA(n_components=30)
    pca_result = pca_reducer.fit_transform([dp for dp in vectors])
    for pr in pca_result:
      yield pr

  def umap(self, vectors: List[NDArray[np.float64]]) -> Generator[NDArray[np.float64], None, None]:
    umap_reducer = UMAP(n_components=30, random_state=42, n_jobs=-1)
    umap_result = umap_reducer.fit_transform([dp for dp in vectors])
    for ur in umap_result:
      yield ur
