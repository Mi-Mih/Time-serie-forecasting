import numpy as np
import pandas as pd

from typing import Tuple

from model.time_serie import TimeSerie


class DecomposeModel:
    def __init__(self, lag: int = 7):
        self.type_ts_decomposition = None
        self.residue = None
        self.lag = lag

        self.freq = None
        self.amplitude = None
        self.seasonality = None
        self.intercept = None

        self.trend_weights = None

        self.steps = 100
        self.ub_freqs = 10
        self.lb_freqs = -10

    @staticmethod
    def get_x_y_arrays(y_axis: pd.Series) -> Tuple[np.array, np.array]:
        x = np.arange(0, y_axis.shape[0])
        y = y_axis.fillna(0).to_numpy()

        return x, y

    def fit_seasonality(self, ts: TimeSerie) -> None:
        # amplitude is const
        # average frequency
        x, y = self.get_x_y_arrays(ts.seasonality)

        # TODO: choose better target func
        # start solution
        if y[1] >= y[0]:
            self.seasonality = lambda x: self.amplitude * np.sin(self.freq * x) + self.intercept
        else:
            self.seasonality = lambda x: self.amplitude * np.cos(self.freq * x) + self.intercept

        self.amplitude = max(y)

        # TODO: define bounds or use gradient descent
        freqs = np.linspace(self.lb_freqs, self.ub_freqs, self.steps)
        intercepts = np.linspace(min(y), max(y), self.steps)

        combinations = dict()
        for freq in freqs:
            for intercept in intercepts:
                self.freq = freq
                self.intercept = intercept

                combinations[(freq, intercept)] = np.mean(np.abs(y - self.seasonality(x)))

        self.freq = min(combinations, key=combinations.get)[0]
        self.intercept = min(combinations, key=combinations.get)[1]

    def fit_trend(self, ts: TimeSerie):

        x, y = self.get_x_y_arrays(ts.trend)

        type_trend = ts.growth

        if type_trend == 'flat':
            self.trend_weights = [np.mean(y)]
        elif type_trend == 'quadric':
            self.trend_weights = np.polyfit(x, y, 2)[::-1]
        elif type_trend == 'linear':
            self.trend_weights = np.polyfit(x, y, 1)[::-1]
        else:
            self.trend_weights = np.polyfit(x, y, 1)[::-1]

    def fit_residue(self, ts: TimeSerie) -> None:
        """
        naive fitting
        """
        _, y = self.get_x_y_arrays(ts.residue)
        self.residue = y[-self.lag:]

    def fit(self, ts: TimeSerie) -> None:

        # generate naive fit for residue
        self.fit_residue(ts)

        # generate season func
        self.fit_trend(ts)

        # generate trend coeffs
        self.fit_seasonality(ts)

        self.type_ts_decomposition = ts.type_decomposition

    def predict_trend(self, x: np.array) -> np.array:
        sum = 0
        for i in range(len(self.trend_weights)):
            sum += self.trend_weights[i] * (x ** i)
        return sum

    def predict(self, x: np.array) -> np.array:
        if self.type_ts_decomposition == 'additive':
            return self.predict_trend(x) + self.seasonality(x) + self.residue
        elif self.type_ts_decomposition == 'multiplicative':
            return self.predict_trend(x) * self.seasonality(x) * self.residue
        else:
            return self.predict_trend(x) + self.seasonality(x) + self.residue


if __name__ == '__main__':
    ...
