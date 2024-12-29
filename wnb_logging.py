import wandb
import pandas as pd

from plot_results import plot_results, plot_feature_importances

FEATURE_NAMES = [
    'intensity_feature',
    'coordinates_feature',
    'gradient_intensity_feature',
    'texture_contrast_feature',
    'texture_correlation_feature',
    'texture_dissimilarity_feature',
    # 'texture',
    't2_features',
    'edge_feature'
]


def log_metric_in_wnb(model, feature_names, result_summary_file, pre_process_params, subfolder_path):
    wandb.init(project="mialab", config=model.get_params())

    results_df = pd.read_csv(result_summary_file, sep=";")
    results_df["KEY"] = results_df.apply(lambda row: f'{row["STATISTIC"]}_{row["METRIC"]}_{row["LABEL"]}', axis=1)
    dice_dict = results_df[["KEY", "VALUE"]].set_index("KEY")["VALUE"].to_dict()

    dice_dict["MEAN_DICE"] = results_df[(results_df["STATISTIC"] == "MEAN") & (results_df["METRIC"] == "DICE")][
        "VALUE"].mean()
    dice_dict["MEAN_HDRFDST"] = results_df[(results_df["STATISTIC"] == "MEAN") & (results_df["METRIC"] == "HDRFDST")][
        "VALUE"].mean()

    for metric in dice_dict.keys():
        wandb.summary[metric] = dice_dict[metric]

    for param in pre_process_params.keys():
        wandb.summary[param] = pre_process_params[param]

    results_fig = plot_results(subfolder_path, return_fig=True)
    wandb.log({"DICE score": wandb.Image(results_fig)})

    features_importance_fig = plot_feature_importances(model, feature_names, return_fig=True)
    wandb.log({"Features importance": wandb.Image(features_importance_fig)})

    wandb.finish()


def get_run_by_name(api, run_name, entity_name, project_name):
    runs = api.runs(f"{entity_name}/{project_name}")

    run_id = None
    for run in runs:
        if run.name == run_name:
            return run


def get_runs_between_the_timestamps(api, first_valid_run_timestamp, last_valid_run_timestamp, entity, project):
    runs = api.runs(f"{entity}/{project}")

    run_dict_list = []

    for run in runs:
        if "_timestamp" not in run.summary.keys():
            continue
        timestamp = run.summary["_timestamp"]
        if (timestamp >= first_valid_run_timestamp) and (timestamp <= last_valid_run_timestamp):
            summary = run.summary._json_dict
            config = {k: v for k, v in run.config.items() if not k.startswith("_")}

            run_dict_list.append({
                "run_id": run.id,
                "run_name": run.name,
                **summary,
                **config
            })

    runs_df = pd.DataFrame(run_dict_list)
    return runs_df


def count_features(row):
    features_count = 0
    for f in FEATURE_NAMES:
        if f != "t2_features":
            features_count += row[f]
    if row["t2_features"]:
        features_count *= 2
    return features_count


def download_df_between_the_timestamps(first_valid_run_name, last_valid_run_name):
    api = wandb.Api()
    entity, project = "alicja-krzeminska-sciga-dl", "mialab"

    first_valid_run = get_run_by_name(api, first_valid_run_name, entity, project)
    first_valid_run_timestamp = first_valid_run.summary["_timestamp"]

    last_valid_run = get_run_by_name(api, last_valid_run_name, entity, project)
    last_valid_run_timestamp = last_valid_run.summary["_timestamp"]

    runs_df = get_runs_between_the_timestamps(api, first_valid_run_timestamp, last_valid_run_timestamp, entity, project)

    metric_columns = [c for c in runs_df.columns if "_DICE" in c or "_HDRFDST" in c]
    interesting_columns = metric_columns + FEATURE_NAMES + ["n_estimators", "texture_window", "_timestamp"]

    runs_df = runs_df.sort_values(by="_timestamp", ascending=False)
    runs_df = runs_df[interesting_columns]

    runs_df = runs_df.drop_duplicates(subset=FEATURE_NAMES + ["n_estimators", "texture_window"], keep='first')

    runs_df["features_number"] = runs_df.apply(lambda row: count_features(row), axis=1)

    return runs_df, metric_columns
