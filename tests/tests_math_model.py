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

        dict_metrics = dict()
        if f"metrics_folder_{dataset}.csv" in os.listdir(f"../output_data/test_data/metrics/"):
            metrics = pd.read_csv(f"../output_data/test_data/metrics/metrics_folder_{dataset}.csv")
        else:
            metrics = pd.DataFrame()

        for target in targets:

            prediction = (prediction
            .loc[(prediction["target"] == target)]
            )

            test = (test
            .loc[test["target"] == target
                 & (test["date"].isin(prediction["date"]))]
            )

            if not prediction['value'].shape[0]:
                print(f"Empty prediction of {target} time serie in dataset {dataset}!")
                continue

            current_mse = mean_squared_error(test['value'], prediction['value'])
            current_mape = mean_absolute_percentage_error(test['value'], prediction['value']) * 100
            current_mae = mean_absolute_error(test['value'], prediction['value'])

            print(f"Results for dataset № {dataset}, target № {target}")
            print(f"MSE = {current_mse}")
            print(f"MAPE = {current_mape} %")
            print(f"MAE = {current_mae}")

            if not metrics.empty:
                print(f"\nLast metrics for dataset № {dataset}, target № {target} found!\n")

                last_mse = metrics[metrics["target"] == target]['MSE'].iloc[0]
                last_mape = metrics[metrics["target"] == target]["MAPE"].iloc[0]
                last_mae = metrics[metrics["target"] == target]["MAE"].iloc[0]

                print(f"Changes for dataset № {dataset}, target № {target}")
                print(f"MSE changed by {last_mse - current_mse}")
                print(f"MAPE changed by {last_mape - current_mape}")
                print(f"MAE changed by {last_mae - current_mae}")

            else:
                print(f"Last metrics for dataset № {dataset}, target № {target} not found!")

            dict_metrics["target"] = target
            dict_metrics.setdefault("MSE", []).append(current_mse)
            dict_metrics.setdefault("MAPE", []).append(current_mape)
            dict_metrics.setdefault("MAE", []).append(current_mae)

        pd.DataFrame(dict_metrics).to_csv(f"../output_data/test_data/metrics/metrics_folder_{dataset}.csv", index=False)


if __name__ == '__main__':
    test_model_accuracy(horizon=7, chosen_model="decompose_model")
