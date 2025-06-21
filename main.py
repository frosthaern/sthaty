from modules.classes.CodeBertTokenizerEncode import CodeBertTokenizeEncode
from modules.classes.Jsonl import Jsonl

if __name__ == "__main__":
  data = Jsonl("dataset.jsonl")
  encodings = CodeBertTokenizeEncode(data)
  # take encodings and extract attention heads
  # and save it in a file ig, i am not sure
