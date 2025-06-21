from modules.classes.CodeBertTokenizerEncode import CodeBertTokenizeEncode
from modules.classes.Jsonl import Jsonl

if __name__ == "__main__":
  data = Jsonl("dataset.jsonl")
  encodings = CodeBertTokenizeEncode(data)
  for e in encodings:
    print(e["input_ids"])
