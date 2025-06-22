from modules.classes.CodeBertTokenizerEncode import CodeBertTokenizeEncode
from modules.classes.Jsonl import Jsonl
from modules.classes.AttentionExtractor import AttentionExtractor
from modules.classes.AttentionAnalyzer  import AttentionAnalyzer

if __name__ == "__main__":
  data = Jsonl("dataset.jsonl", max_lines=5)
  encodings = CodeBertTokenizeEncode(data)
  attentions = AttentionExtractor(encodings)
  analyzer = AttentionAnalyzer(attentions)
  analyzer.print_attention_shape()
  # take encodings and extract attention heads
  # and save it in a file ig, i am not sure
