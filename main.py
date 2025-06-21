from modules.Jsonl import Jsonl
from modules.CodeBertTokenizer import CodeBertTokenizer

if __name__ == '__main__':
	data = Jsonl('dataset.jsonl')
	tokenizer = CodeBertTokenizer(data)
