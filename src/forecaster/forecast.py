import pandas as pd
import datetime

from src.model.time_serie import TimeSerie
from typing import Dict, Tuple
from src.math_model.decompose_model import DecomposeModel
import numpy as np

class Forecaster:
    def __init__(self, time_series: Dict[Tuple[int, ...], TimeSerie], horizon: int = 1,
                     chosen_model: str = 'decompose_model'):

        self.time_series = time_series
        self.horizon = horizon
        self.chosen_model = chosen_model

    def get_forecast(self) -> Dict[Tuple[int, ...], Dict[str, np.array]]:
        lags = self.horizon
        only_forecasts = dict()

        models = {'decompose_model': DecomposeModel(lag=lags)}

        for name_ts, ts in self.time_series.items():

            if ts.data["value"].shape[0] < self.horizon:
                lags -= ts.data["value"].shape[0]

            model = models.get(self.chosen_model, DecomposeModel(lag=lags))
            model.fit(ts)

            prediction_real_interval = pd.date_range(
                start=pd.to_datetime(ts.data['date'][-1]) + datetime.timedelta(days=1),
                periods=self.horizon)

            prediction_model_interval = np.arange(ts.data['value'].shape[0], ts.data['value'].shape[0] + self.horizon)
            prediction = model.predict(prediction_model_interval)

            # create train + prediction ts
            self.time_series[name_ts].data['value'] = np.append(self.time_series[name_ts].data['value'], prediction)
            self.time_series[name_ts].data['date'] = np.append(self.time_series[name_ts].data['date'], prediction_real_interval)

            # save only prediction part
            only_forecasts[name_ts] = {'value': prediction, 'date': prediction_real_interval}

        return only_forecasts







if __name__ == '__main__':
    ...
