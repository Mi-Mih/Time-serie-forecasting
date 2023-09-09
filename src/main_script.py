import pandas as pd
from src.modes.validation import Validation
from src.model.ts_fabric import TimeSerieFabric
from src.forecaster.forecast import Forecaster
from src.utils.collect_data import write_to_excel


def main(input_path: str, output_path: str, strict_mode: int = 1, horizon: int = 1,
         chosen_model: str = 'decompose_model') -> None:
    data = pd.read_excel(input_path)

    if strict_mode:
        validator = Validation(data)
        validator.validate()

        data = validator.get_correct_data()

    time_series_box = TimeSerieFabric.create(data)

    forecaster = Forecaster(time_series=time_series_box, horizon=horizon, chosen_model=chosen_model)
    forecasts = forecaster.get_forecast()

    write_to_excel(output_path=output_path, only_forecasts=forecasts, time_series=time_series_box)


if __name__ == '__main__':
    ts_folder = 1
    horizon = 7
    main(input_path=f"../input_data/{ts_folder}/{ts_folder}.xlsx",
         output_path=f"../output_data/forecasts_{ts_folder}.xlsx",
         horizon=horizon,
         chosen_model="decompose_model")
