import pandas as pd
import datetime

from src.model.time_serie import TimeSerie
from typing import Dict, Tuple
from src.math_model.decompose_model import DecomposeModel
import numpy as np


def collect_to_df(ts_dict: Dict, writer: pd.ExcelWriter, sheet_name: str, ) -> pd.DataFrame:
    output_df = pd.DataFrame()
    for name_ts, ts in ts_dict.items():

        if isinstance(ts, dict):
            df = pd.DataFrame({'date': ts['date'], 'value': ts['value']})
        else:
            df = pd.DataFrame({'date': ts.data['date'], 'value': ts.data['value']})

        for index_name, value_column in enumerate(name_ts):
            df['target'] = value_column
        output_df = pd.concat([output_df, df], ignore_index=True)

    output_df.to_excel(writer, sheet_name=sheet_name)


def get_forecast(time_series: Dict[Tuple, TimeSerie], horizon: int = 1, output_path: str = '', chosen_model: str = 'decompose_model') -> None:
    lags = horizon
    only_forecasts = dict()

    models = {'decompose_model': DecomposeModel(lag=lags)}

    for name_ts, ts in time_series.items():

        if ts.data["value"].shape[0] < horizon:
            lags -= ts.data["value"].shape[0]

        model = models.get(chosen_model, DecomposeModel(lag=lags))
        model.fit(ts)

        prediction_real_interval = pd.date_range(start=pd.to_datetime(ts.data['date'][-1]) + datetime.timedelta(days=1),
                                                 periods=horizon)

        prediction_model_interval = np.arange(ts.data['value'].shape[0], ts.data['value'].shape[0] + horizon)
        prediction = model.predict(prediction_model_interval)

        # create train + prediction ts
        time_series[name_ts].data['value'] = np.append(time_series[name_ts].data['value'], prediction)
        time_series[name_ts].data['date'] = np.append(time_series[name_ts].data['date'], prediction_real_interval)

        # save only prediction part
        only_forecasts[name_ts] = {'value': prediction, 'date': prediction_real_interval}

    writer = pd.ExcelWriter(output_path, engine='openpyxl')

    collect_to_df(writer=writer,
                  ts_dict=only_forecasts,
                  sheet_name='forecasts')
    collect_to_df(writer=writer,
                  ts_dict=time_series,
                  sheet_name='full_ts')

    writer.close()


if __name__ == '__main__':
    ...
