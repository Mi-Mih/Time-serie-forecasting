import numpy as np
import pandas as pd
from typing import Dict, Tuple
from model.time_serie import TimeSerie

def write_to_excel(output_path: str, only_forecasts: Dict[Tuple, Dict[str, np.array]], time_series: Dict[Tuple, TimeSerie]) -> None:
    writer = pd.ExcelWriter(output_path, engine='openpyxl')

    collect_to_df(writer=writer,
                  ts_dict=only_forecasts,
                  sheet_name='forecasts')
    collect_to_df(writer=writer,
                  ts_dict=time_series,
                  sheet_name='full_ts')

    writer.close()

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
