from modules.classes.CodeBertTokenizerEncode import CodeBertTokenizeEncode
from modules.classes.Jsonl import Jsonl
from modules.classes.AttentionExtractor import AttentionExtractor
from modules.classes.AttentionAnalyzer  import AttentionAnalyzer
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-n", help="no of codepoints to take")
  args = parser.parse_args()
  data = Jsonl("dataset.jsonl", max_lines=int(args.n))
  encodings = CodeBertTokenizeEncode(data)
  attentions = AttentionExtractor(encodings)
  analyzer = AttentionAnalyzer(attentions)
  entropy_values = [en.flatten() for en in analyzer.entropy()]
  pca_result = [pr for pr in analyzer.pca(entropy_values)]
  umap_result = [ur for ur in analyzer.umap(pca_result)]
  print(f"{len(umap_result)}")
  print(f"{umap_result[0].shape}")
    
