from collections import defaultdict
from typing import Callable, List

import numpy as np
import orjson as json
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class DataSet:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data: List[str] = []

    def load_data(self, conversion_function: Callable[[str], List[str]]):
        self.data = conversion_function(self.file_path)


def convert_jsonl_to_list_str(jsonl_file_path: str, max_lines: int = 50) -> List[str]:
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
            model, attn_implementation="eager"
        )
        self.texts: List[str] = texts

    def tokenize(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def encode(self, texts: List[str]) -> dict:
        encoded = defaultdict(list)
        for text in texts:
            encoded_ = self.tokenizer.encode_plus(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512,
            )
            for k, v in encoded_.items():
                encoded[k].append(v)
        return encoded

    def _process_batch(self, texts: List[str]) -> np.ndarray:
        encoded = self.encode(texts)

        device = next(self.model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_attentions=True,
            )
            batch_attentions = np.stack(
                [
                    np.stack([attn.detach().cpu().numpy() for attn in layer_attentions])
                    for layer_attentions in zip(*outputs.attentions)
                ],
                axis=1,
            )

        return batch_attentions

    def get_attention_heads(self, batch_size: int = 8) -> np.ndarray:
        all_attentions = []
        num_batches = (len(self.texts) + batch_size - 1) // batch_size

        with tqdm(
            total=len(self.texts),
            desc="Processing batches",
            unit="text",
            dynamic_ncols=True,
        ) as pbar:
            for i in range(0, len(self.texts), batch_size):
                batch_texts = self.texts[i : i + batch_size]
                batch_attentions = self._process_batch(batch_texts)
                all_attentions.append(batch_attentions)
                pbar.update(len(batch_texts))

                pbar.set_postfix(
                    {
                        "batch": f"{i // batch_size + 1}/{num_batches}",
                        "batch_size": len(batch_texts),
                    }
                )

        return np.concatenate(all_attentions, axis=0)


def main():
    dataset = DataSet("dataset.jsonl")
    dataset.load_data(lambda x: convert_jsonl_to_list_str(x, max_lines=5))

    tokenizer = Tokenizer("microsoft/codebert-base", dataset.data)
    attention_heads = tokenizer.get_attention_heads()

    print(attention_heads.shape)


if __name__ == "__main__":
    main()
