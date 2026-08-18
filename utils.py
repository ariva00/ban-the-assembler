import os
import random

import numpy
import torch

def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_model_dtype(model):
    return next(model.parameters()).dtype

class CSVLogger:
    def __init__(self, filename, columns=list[str], append=True, sep=","):
        self.filename = filename
        self.columns = columns
        self.sep = sep

        if append and os.path.exists(filename):
            with open(filename, "r") as r:
                first_line = r.readline().replace("\n", "")
                file_columns = first_line.split(sep)
                assert all([col in file_columns for col in columns]) and all([col in columns for col in file_columns]), f"Columns in {filename} {file_columns} do not match {columns}"
                self.columns = file_columns # to keep order of columns consistent with file
        else:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                f.write(sep.join(columns) + "\n")

    def write(self, data):
        assert len(data) == len(self.columns), f"Data should match number of columns {self.columns}"
        assert all([str(d).find(self.sep) == -1 for d in data]), f"Separator '{self.sep}' found in data, can't write data"
        with open(self.filename, "a") as f:
            data = [str(d) for d in data]
            f.write(self.sep.join(data) + "\n")

    def get_num_rows(self, exclude_header=True):
        with open(self.filename, "r") as f:
            return len(f.readlines()) - 1 if exclude_header else len(f.readlines())

if __name__ == '__main__':
    print(CSVLogger('test.csv', ['one', 'two', 'three']).get_num_rows())