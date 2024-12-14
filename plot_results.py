import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px


def plot_results(subfolder_path: str, return_fig: bool = False):
    csv_path = os.path.join(subfolder_path, 'results.csv')

    if os.path.exists(csv_path):
        try:
            data = pd.read_csv(csv_path, delimiter=';')
        except Exception as e:
            print(f"An error occurred while loading '{csv_path}': {e}")
            return

        # Ensure that the relevant columns exist
        if not {'LABEL', 'DICE'}.issubset(data.columns):
            print(
                f"Error: The required columns 'LABEL' and 'DICE' do not exist in the data for '{csv_path}'.")
            return

        # Group the data by label and extract the Dice coefficients
        labels = ["WhiteMatter", "GreyMatter", "Hippocampus", "Amygdala", "Thalamus"]
        label_data = [data[data['LABEL'] == label]['DICE'].dropna() for label in labels]

        # Plot the Dice coefficients per label in a boxplot
        fig = plt.figure()
        plt.boxplot(label_data, labels=labels)
        plt.xlabel('Labels')
        plt.ylabel('Dice Coefficients')
        plt.title(f'Boxplot of Dice Coefficients per Label for {os.path.basename(subfolder_path)}')

        if return_fig:
            return fig
        else:
            plt.show()


def plot_feature_importances(forest: RandomForestClassifier, feature_names: list, return_fig: bool = True):
    importances = forest.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig = plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if return_fig:
        return fig
    else:
        plt.show()


def plot_scores(runs_df: pd.DataFrame, n_estimators: int, texture_window: int, metric_columns: list):
    runs_df = runs_df[(runs_df["n_estimators"] == n_estimators) & (runs_df["texture_window"] == texture_window)]

    mean_columns = [c for c in metric_columns if "MEAN_" in c and c in runs_df.columns]

    df_long = runs_df.melt(
        id_vars=['features_number', 'intensity_feature'],
        value_vars=mean_columns,
        var_name='Feature name',
        value_name='value'
    )

    fig = px.scatter(
        df_long,
        x='features_number',
        y='value',
        color='Feature name',
        labels={'features_number': '#features', 'value': 'Value'},
        title=f'Metrics values with respect to features number (n_estimators={n_estimators}, texture_window: {texture_window})',
        trendline='ols',
    )

    traces_to_update = fig.data
    for trace in traces_to_update:
        highlighted = df_long[(df_long['Feature name'] == trace.name) & (df_long['intensity_feature'] == 1)]

        if not highlighted.empty:
            fig.add_scatter(
                x=highlighted['features_number'],
                y=highlighted['value'],
                mode='markers',
                marker=dict(size=12, symbol='circle-open', color="gray"),
                name=trace.name,
                legendgroup=trace.name,
                showlegend=False
            )

    fig.show()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mia_result_dir = os.path.join(script_dir, 'mia-result')

    try:
        # Iterate through each subfolder in the mia-result directory
        for subfolder in os.listdir(mia_result_dir):
            subfolder_path = os.path.join(mia_result_dir, subfolder)
            plot_results(subfolder_path)

    except FileNotFoundError:
        print("Error: The 'mia-result' directory could not be found.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return


if __name__ == '__main__':
    main()
