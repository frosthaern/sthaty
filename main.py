from typing import Callable, List
import orjson as json


class DataSet:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data: List[str] = []

    def load_data(self, conversion_function: Callable[[str], List[str]]):
        self.data = conversion_function(self.file_path)


def convert_jsonl_to_list_str(jsonl_file_path: str) -> List[str]:
    with open(jsonl_file_path, "r") as f:
        res: List[str] = []
        for line in f:
            data = json.loads(line)
            res.append(f"{data['query']}{data['code']}")
        return res


def main():
    dataset = DataSet("dataset.json")
    dataset.load_data(convert_jsonl_to_list_str)
    print(len(dataset.data))


if __name__ == "__main__":
    main()
