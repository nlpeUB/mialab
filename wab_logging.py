import wandb
import pandas as pd


def log_metric_in_wab(model, result_summary_file):
    wandb.init(project="mialab", config=model.get_params())

    results_df = pd.read_csv(result_summary_file, sep=";")
    results_df["KEY"] = results_df.apply(lambda row: f'{row["STATISTIC"]}_{row["METRIC"]}_{row["LABEL"]}', axis=1)
    dice_dict = results_df[["KEY", "VALUE"]].set_index("KEY")["VALUE"].to_dict()

    dice_mean = results_df[results_df["STATISTIC"] == "MEAN"]["VALUE"].mean()
    dice_dict["MEAN_DICE"] = dice_mean

    for metric in dice_dict.keys():
        wandb.summary[metric] = dice_dict[metric]

    wandb.finish()
