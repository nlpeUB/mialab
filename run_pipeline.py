from pipeline import main
from itertools import product
from tqdm import tqdm


def run_pipeline():
    result_dir = './mia-result'
    data_atlas_dir = '../data/atlas'
    data_train_dir = '../data/train/'
    data_test_dir = '../data/test/'

    fixed_feature_extraction_params = {
        'load_features': True,
        'save_features': False,
        'overwrite': False,
    }

    binary_feature_extraction_params = {
        'intensity_feature': True,
        'coordinates_feature': True,
        'gradient_intensity_feature': True,
        'texture_contrast_feature': True,
        'texture_dissimilarity_feature': True,
        'texture_correlation_feature': True,
        'gradient_intensity_feature': True,
        't2_features': True,
        'edge_feature': True
    }

    feature_extraction_params = {**fixed_feature_extraction_params, **binary_feature_extraction_params}

    keys = list(feature_extraction_params.keys())
    all_combinations = list(product([True, False], repeat=len(keys)))

    for feature_extraction_params_ in tqdm(all_combinations):
        if sum(feature_extraction_params_) == 0:
            continue
        current_params = dict(zip(keys, feature_extraction_params_))
        print(f"Running with parameters: {current_params}")

        main(result_dir, data_atlas_dir, data_train_dir, data_test_dir, feature_extraction_params)


if __name__ == "__main__":
    run_pipeline()
