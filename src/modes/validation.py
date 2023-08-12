import pandas as pd


class Validation:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.rules = []
        self.true_column_names = {'date', 'value'}

    def check_empty_data(self) -> None:
        if self.data.empty:
            self.rules.append("EMPTY INPUT DATA")

    def check_name_columns(self) -> None:
        column_names = set(self.data.columns)

        if len(column_names.intersection(self.true_column_names)) < len(self.true_column_names):
            self.rules.append("REQUIRED COLUMN MISSING")

        duplicates_columns = [column for column in self.data.columns if column in self.true_column_names]
        if len(duplicates_columns) != len(self.true_column_names):
            self.rules.append("DUPLICATES COLUMNS")

    def check_type_data(self):

        numeric_value = self.data[pd.to_numeric(self.data["value"], errors='coerce').notnull()]['value']
        if numeric_value.shape[0] < self.data["value"].shape[0]:
            self.rules.append('Incorrect type of data in column "value"')

        dates = pd.to_datetime(self.data["date"], errors='coerce')
        if dates.isnull().sum():
            self.rules.append('Incorrect type of data in column "date"')

    def convert_type_data(self):
        self.data["date"] = pd.to_datetime(self.data["date"])
        self.data["value"] = pd.to_numeric(self.data["value"])

    def validate(self) -> None:

        self.check_empty_data()
        self.check_name_columns()
        self.check_type_data()

        for rule in self.rules:
            print(rule)

        for rule in self.rules:
            print(f"Expection: {rule}")
            raise Exception("Incorrect input data")

        self.convert_type_data()

    def get_correct_data(self):
        return self.data


if __name__ == '__main__':
    ...
