import wandb
import pandas as pd

from plot_results import plot_results


def log_metric_in_wab(model, result_summary_file, pre_process_params, subfolder_path):
    wandb.init(project="mialab", config=model.get_params())

    results_df = pd.read_csv(result_summary_file, sep=";")
    results_df["KEY"] = results_df.apply(lambda row: f'{row["STATISTIC"]}_{row["METRIC"]}_{row["LABEL"]}', axis=1)
    dice_dict = results_df[["KEY", "VALUE"]].set_index("KEY")["VALUE"].to_dict()

    dice_mean = results_df[results_df["STATISTIC"] == "MEAN"]["VALUE"].mean()
    dice_dict["MEAN_DICE"] = dice_mean

    for metric in dice_dict.keys():
        wandb.summary[metric] = dice_dict[metric]

    for param in pre_process_params.keys():
        wandb.summary[param] = pre_process_params[param]

    results_fig = plot_results(subfolder_path, return_fig=True)
    wandb.log({"plot": wandb.Image(results_fig)})

    wandb.finish()