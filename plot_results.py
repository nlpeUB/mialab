import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def main():
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus)
    #  in a boxplot

    try:

        script_dir = os.path.dirname(os.path.abspath(__file__))
        mia_result_dir = os.path.join(script_dir, 'mia-result')

        # Iterate through each subfolder in the mia-result directory
        for subfolder in os.listdir(mia_result_dir):
            subfolder_path = os.path.join(mia_result_dir, subfolder)
            if os.path.isdir(subfolder_path):
                csv_path = os.path.join(subfolder_path, 'results.csv')


                if os.path.exists(csv_path):
                    try:
                        data = pd.read_csv(csv_path, delimiter=';')
                    except Exception as e:
                        print(f"An error occurred while loading '{csv_path}': {e}")
                        continue

                    # Ensure that the relevant columns exist
                    if not {'LABEL', 'DICE'}.issubset(data.columns):
                        print(f"Error: The required columns 'LABEL' and 'DICE' do not exist in the data for '{csv_path}'.")
                        continue

                    # Group the data by label and extract the Dice coefficients
                    labels = ["WhiteMatter", "GreyMatter", "Hippocampus", "Amygdala", "Thalamus"]
                    label_data = [data[data['LABEL'] == label]['DICE'].dropna() for label in labels]

                    # Plot the Dice coefficients per label in a boxplot
                    plt.figure()
                    plt.boxplot(label_data, labels=labels)
                    plt.xlabel('Labels')
                    plt.ylabel('Dice Coefficients')
                    plt.title(f'Boxplot of Dice Coefficients per Label for {subfolder}')
                    plt.show()
    except FileNotFoundError:
        print("Error: The 'mia-result' directory could not be found.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return


if __name__ == '__main__':
    main()
