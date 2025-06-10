from typing import Callable, List


class DataSet:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()

    def load_data(self, conversion_function: Callable[[str], List[str]]):
        with open(self.file_path, "r") as f:
            self.data = conversion_function(f)


def main():
    pass


if __name__ == "__main__":
    main()
