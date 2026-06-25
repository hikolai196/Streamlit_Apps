import pandas as pd


class DataManager:
    def __init__(self):
        self.df = None

    def load_file(self, file):
        if file.name.endswith(".csv"):
            self.df = pd.read_csv(file)
        else:
            self.df = pd.read_excel(file)

    def get_preview(self):
        return self.df.head()

    def get_schema(self):
        return str(self.df.dtypes)
