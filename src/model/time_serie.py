import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Dict
import matplotlib.pyplot as plt
import pymannkendall as mk


class TimeSerie:
    def __init__(self, data: Dict[str, np.array] = None, period_decomposition: int = 2,
                 type_decomposition: str = 'additive'):
        """
        :param data: dictionary with dates in key and values of time serie in value
        """
        self.growth = None
        self.data = data

        self.seasonality = None
        self.trend = None
        self.residue = None

        self.period_decomposition = period_decomposition
        self.type_decomposition = type_decomposition

        if isinstance(data, dict):
            self.decompose(data, period_decomposition, type_decomposition)
            self.get_type_trend(data['value'])

    def get_type_trend(self, serie: np.array) -> None:
        """
        Mann-Kendall test for trend monotonicity
        """
        type_trend = mk.original_test(serie).trend

        if type_trend == 'no trend':
            self.growth = 'flat'
        else:
            self.growth = 'linear'

    def decompose(self, data: pd.DataFrame = None, period_decomposition: int = 2,
                  type_decomposition: str = 'additive') -> None:
        if data:
            decompose_result = seasonal_decompose(
                pd.DataFrame({'value': data['value']}, index=pd.to_datetime(data['date'])), model=type_decomposition,
                period=period_decomposition)

            self.seasonality = decompose_result.seasonal
            self.trend = decompose_result.trend
            self.residue = decompose_result.resid
        else:
            decompose_result = seasonal_decompose(
                pd.DataFrame({'value': self.data['value']}, index=pd.to_datetime(self.data['date'])),
                model=self.type_decomposition,
                period=self.period_decomposition)

            self.seasonality = decompose_result.seasonal.fillna(decompose_result.seasonal.mean())
            self.trend = decompose_result.trend.fillna(decompose_result.seasonal.mean())
            self.residue = decompose_result.resid.fillna(decompose_result.seasonal.mean())


    def from_df(self, df: pd.DataFrame, name_date_column='date', name_value_column='value') -> Dict[str, np.array]:
        date = df[name_date_column].to_numpy()
        value = df[name_value_column].to_numpy()

        self.data = {'date': date, 'value': value}
        self.decompose(period_decomposition=self.period_decomposition, type_decomposition=self.type_decomposition)
        self.get_type_trend(self.data['value'])

    def from_numpy(self, date: np.array, value: np.array) -> Dict[str, np.array]:
        self.data = {'date': date, 'value': value}
        self.decompose(period_decomposition=self.period_decomposition, type_decomposition=self.type_decomposition)
        self.get_type_trend(self.data['value'])

    def from_list(self, date: list, value: list) -> Dict[str, np.array]:
        self.data = {'date': np.array(date), 'value': np.array(value)}
        self.decompose(period_decomposition=self.period_decomposition, type_decomposition=self.type_decomposition)
        self.get_type_trend(self.data['value'])

    def plot_ts(self, color='blue') -> None:
        fig, ax = plt.subplots()
        ax.plot(self.data['date'], self.data['value'], color=color, label='time serie')
        ax.xlabel('date')
        ax.ylabel('value')
        ax.grid()
        ax.legend()
        plt.show()

    def plot_parts(self, color='blue') -> None:
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        ax1.plot(self.data['date'], self.seasonality, color=color, label='seasonality')
        ax2.plot(self.data['date'], self.trend, color=color, label='trend')
        ax3.plot(self.data['date'], self.residue, color=color, label='residue')

        ax1.xlabel('date')
        ax1.ylabel('value')
        ax1.grid()
        ax1.legend()
        ax2.xlabel('date')
        ax2.ylabel('value')
        ax2.grid()
        ax2.legend()
        ax3.xlabel('date')
        ax3.ylabel('value')
        ax3.grid()
        ax3.legend()
        plt.show()


if __name__ == '__main__':
    ts = TimeSerie({'date': np.array(['01-01-2020', '02-01-2020', '03-01-2020']), 'value': np.array([1, 2, 3])}, 1)
