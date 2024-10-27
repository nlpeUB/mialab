from pipeline import main
from itertools import product


def run_pipeline():
    result_dir = './mia-result'
    data_atlas_dir = '../data/atlas'
    data_train_dir = '../data/train/'
    data_test_dir = '../data/test/'

    feature_extraction_params = {
        'coordinates_feature': False,
        'intensity_feature': False,
        'gradient_intensity_feature': False,
        'neighborhood_features': False,
        'texture_features': False,
        'edge_features': False
    }

    keys = list(feature_extraction_params.keys())
    all_combinations = list(product([True, False], repeat=len(keys)))

    for feature_extraction_params_ in all_combinations:
        current_params = dict(zip(keys, feature_extraction_params_))
        print(f"Running with parameters: {current_params}")

        main(result_dir, data_atlas_dir, data_train_dir, data_test_dir, feature_extraction_params)
