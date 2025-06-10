from typing import Callable, List
import orjson as json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class DataSet:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data: List[str] = []

    def load_data(self, conversion_function: Callable[[str], List[str]]):
        self.data = conversion_function(self.file_path)


def convert_jsonl_to_list_str(jsonl_file_path: str, max_lines: int = 50) -> List[str]:
    """Convert JSONL file to list of strings, with optional limit on number of lines"""
    with open(jsonl_file_path, "r") as f:
        res: List[str] = []
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            data = json.loads(line)
            res.append(f"<s>{data['query']}</s>{data['code']}</s>")
        return res


class Tokenizer:
    def __init__(self, model: str, texts: List[str]):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model,
            attn_implementation="eager"
        )
        self.texts: List[str] = texts

    def tokenize(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def encode(self, text: str) -> dict:
        """Encode text and return both input_ids and attention_mask"""
        encoded = self.tokenizer.encode_plus(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        return encoded

    def get_attention_heads(self) -> List[torch.Tensor]:
        res = []
        for text in self.texts:
            encoded = self.encode(text)
            outputs = self.model(
                encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                output_attentions=True
            )
            res.append(outputs.attentions)
        return res


def main():
    dataset = DataSet("dataset.jsonl")
    dataset.load_data(lambda x: convert_jsonl_to_list_str(x, max_lines=5))

    tokenizer = Tokenizer("microsoft/codebert-base", dataset.data)
    attention_heads = tokenizer.get_attention_heads()

    print(len(attention_heads))


if __name__ == "__main__":
    main()
