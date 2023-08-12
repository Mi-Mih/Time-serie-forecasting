from main_script import main

import pandas as pd
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def test_model_accuracy(horizon: int = 1, chosen_model: str = 'decompose_model'):

    datasets = os.listdir(f"../input_data/test_datasets/train")

    for dataset in datasets:

        main(input_path=f"../input_data/test_datasets/train/{dataset}",
             output_path=f"../output_data/test_data/forecasts_horizon_{horizon}_folder_{dataset}",
             horizon=horizon,
             chosen_model=chosen_model)

        test = pd.read_excel(f"../input_data/test_datasets/test/{dataset}")
        prediction = pd.read_excel(f"../output_data/test_data/forecasts_horizon_{horizon}_folder_{dataset}",
                                   sheet_name='forecasts')

        targets = test["target"].unique()

        for target in targets:

            prediction = (prediction
                .loc[(prediction["target"] == target)]
            )

            test = (test
                .loc[test["target"] == target
                & (test["date"].isin(prediction["date"]))]
            )

            if not prediction['value'].shape[0]:
                print(f"Empty prediction of {target} time serie!")
                continue

            print(f"Results for dataset № {dataset}")
            print(f"MSE = {mean_squared_error(test['value'], prediction['value'])}")
            print(f"MAPE = {mean_absolute_percentage_error(test['value'], prediction['value']) * 100} %")
            print(f"MAE = {mean_absolute_error(test['value'], prediction['value'])}")


if __name__ == '__main__':
    test_model_accuracy(horizon=7, chosen_model="decompose_model")
