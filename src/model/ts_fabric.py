import pandas as pd
from model.time_serie import TimeSerie
from typing import Dict, Tuple


class TimeSerieFabric:
    @staticmethod
    def create(df: pd.DataFrame, period_decomposition: int = 2,
               type_decomposition: str = 'additive') -> Dict[Tuple[int, ...], TimeSerie]:
        """
        convert from pandas.DataFrame with time series to dict with time series
        :param type_decomposition: type_decomposition
        :param period_decomposition: period_decomposition
        :param df: frame of time series
        :return: None
        """
        time_series = {}

        columns_df = list(df.columns)

        if 'value' not in columns_df:
            raise KeyError("REQUIRED COLUMN MISSING value")

        for column in ['date', 'value']:
            if column in columns_df:
                columns_df.remove(column)
            else:
                raise KeyError(f"REQUIRED COLUMN MISSING {column}")

        groups = df.groupby(columns_df)['value'].sum().index
        for index in groups:
            if not isinstance(index, tuple):
                index = (index,)
            for index_column, column in enumerate(columns_df):

                current_df = (df
                .loc[df[column] == index[index_column]]
                )

                if 'date' not in current_df.columns:
                    continue

                ts_data = {'date': current_df['date'].to_numpy(), 'value': current_df['value'].to_numpy()}

                time_series[index] = TimeSerie(data=ts_data,
                                               period_decomposition=period_decomposition,
                                               type_decomposition=type_decomposition)

        return time_series


if __name__ == '__main__':
    ...
