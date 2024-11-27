from pipeline import main
from itertools import product
from tqdm import tqdm


def run_pipeline():
    result_dir = './mia-result'
    data_atlas_dir = '../data/atlas'
    data_train_dir = '../data/train/'
    data_test_dir = '../data/test/'

    fixed_feature_extraction_params = {
        'load_features': False,
        'save_features': True,
        'overwrite': True,
    }

    binary_feature_extraction_params = {
        'intensity_feature': True,
        'coordinates_feature': True,
        'gradient_intensity_feature': True,
        'texture': True,
        't2_features': True,
        'edge_feature': True
    }

    # keys = list(binary_feature_extraction_params.keys())
    # all_combinations = list(product([True, False], repeat=len(keys)))
    #
    # for binary_feature_extraction_params_ in tqdm(all_combinations):
    #     if sum(binary_feature_extraction_params_) == 0:
    #         continue
    #
    #     current_params = dict(zip(keys, binary_feature_extraction_params_))

    current_params = binary_feature_extraction_params

    if current_params["texture"]:
        current_params["texture_contrast_feature"] = True
        current_params["texture_dissimilarity_feature"] = True
        current_params["texture_correlation_feature"] = True
    else:
        current_params["texture_contrast_feature"] = False
        current_params["texture_dissimilarity_feature"] = False
        current_params["texture_correlation_feature"] = False
    del current_params["texture"]

    feature_extraction_params = {**fixed_feature_extraction_params, **current_params}

    print(f"Running with parameters: {current_params}")

    main(result_dir, data_atlas_dir, data_train_dir, data_test_dir, feature_extraction_params)


if __name__ == "__main__":
    run_pipeline()
