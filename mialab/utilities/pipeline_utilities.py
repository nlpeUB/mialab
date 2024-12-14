"""This module contains utility classes and functions."""
import enum
import os
import typing as t
import warnings

import numpy as np
import pymia.data.conversion as conversion
import pymia.filtering.filter as fltr
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.metric as metric
import SimpleITK as sitk

import mialab.data.structure as structure
import mialab.filtering.feature_extraction as fltr_feat
import mialab.filtering.postprocessing as fltr_postp
import mialab.filtering.preprocessing as fltr_prep
import mialab.utilities.multi_processor as mproc

atlas_t1 = sitk.Image()
atlas_t2 = sitk.Image()


def load_atlas_images(directory: str):
    """Loads the T1 and T2 atlas images.

    Args:
        directory (str): The atlas data directory.
    """

    global atlas_t1
    global atlas_t2
    atlas_t1 = sitk.ReadImage(os.path.join(directory, 'mni_icbm152_t1_tal_nlin_sym_09a_mask.nii.gz'))
    atlas_t2 = sitk.ReadImage(os.path.join(directory, 'mni_icbm152_t2_tal_nlin_sym_09a.nii.gz'))
    if not conversion.ImageProperties(atlas_t1) == conversion.ImageProperties(atlas_t2):
        raise ValueError('T1w and T2w atlas images have not the same image properties')


class FeatureImageTypes(enum.Enum):
    """Represents the feature image types."""

    ATLAS_COORD = 1
    T1w_INTENSITY = 2
    T1w_GRADIENT_INTENSITY = 3
    T2w_INTENSITY = 4
    T2w_GRADIENT_INTENSITY = 5
    T1w_TEXTURE_CONTRAST = 6
    T2w_TEXTURE_CONTRAST = 7
    T1w_TEXTURE_DISSIMILARITY = 8
    T2w_TEXTURE_DISSIMILARITY = 9
    T1w_TEXTURE_CORRELATION = 10
    T2w_TEXTURE_CORRELATION = 11
    T1w_EDGES = 12
    T2w_EDGES = 13


class FeatureExtractor:
    """Represents a feature extractor."""

    def __init__(self, img: structure.BrainImage, **kwargs):
        """Initializes a new instance of the FeatureExtractor class.

        Args:
            img (structure.BrainImage): The image to extract features from.
        """
        self.img = img
        self.training = kwargs.get('training', True)
        self.load_features = kwargs.get('load_features', True)
        self.save_features = kwargs.get('save_features', False)
        self.overwrite = kwargs.get('overwrite', False)

        self.t2_features = kwargs.get('t2_features', False)
        self.coordinates_feature = kwargs.get('coordinates_feature', False)
        self.intensity_feature = kwargs.get('intensity_feature', False)
        self.gradient_intensity_feature = kwargs.get('gradient_intensity_feature', False)
        self.texture_contrast_feature = kwargs.get('texture_contrast_feature', False)
        self.texture_dissimilarity_feature = kwargs.get('texture_dissimilarity_feature', False)
        self.texture_correlation_feature = kwargs.get('texture_correlation_feature', False)
        self.edge_feature = kwargs.get('edge_feature', False)

        self.feature_names = []

    def execute(self) -> structure.BrainImage:
        """Extracts features from an image.

        Returns:
            structure.BrainImage: The image with extracted features.
        """
        if self.load_features:
            self._load_features()
        else:
            self._compute_features()

        # We can keep it here because it is not computed/loaded/saved.
        if self.intensity_feature:
            feature_image_type = FeatureImageTypes.T1w_INTENSITY
            self.feature_names.append(feature_image_type.name)
            self.img.feature_images[feature_image_type] = self.img.images[structure.BrainImageTypes.T1w]
            if self.t2_features:
                feature_image_type = FeatureImageTypes.T2w_INTENSITY
                self.feature_names.append(feature_image_type.name)
                self.img.feature_images[feature_image_type] = self.img.images[structure.BrainImageTypes.T2w]

        if self.save_features:
            if self.load_features and not self.overwrite:
                print(f"If `load_features` == True and `overwrite` == False, save_features should be False!")
            else:
                self._save_features()

        self._generate_feature_matrix()
        self.img.feature_names = self.feature_names

        return self.img

    def _load_features(self):
        if self.coordinates_feature:
            self._load_feature(FeatureImageTypes.ATLAS_COORD)

        if self.gradient_intensity_feature:
            self._load_feature(FeatureImageTypes.T1w_GRADIENT_INTENSITY)
            if self.t2_features:
                self._load_feature(FeatureImageTypes.T2w_GRADIENT_INTENSITY)

        if self.texture_contrast_feature:
            self._load_feature(FeatureImageTypes.T1w_TEXTURE_CONTRAST)
            if self.t2_features:
                self._load_feature(FeatureImageTypes.T2w_TEXTURE_CONTRAST)

        if self.texture_dissimilarity_feature:
            self._load_feature(FeatureImageTypes.T1w_TEXTURE_DISSIMILARITY)
            if self.t2_features:
                self._load_feature(FeatureImageTypes.T2w_TEXTURE_DISSIMILARITY)

        if self.texture_correlation_feature:
            self._load_feature(FeatureImageTypes.T1w_TEXTURE_CORRELATION)
            if self.t2_features:
                self._load_feature(FeatureImageTypes.T2w_TEXTURE_CORRELATION)

        if self.edge_feature:
            self._load_feature(FeatureImageTypes.T1w_EDGES)
            if self.t2_features:
                self._load_feature(FeatureImageTypes.T2w_EDGES)

    def _load_feature(self, feature_image_type: FeatureImageTypes):
        feature_name = feature_image_type.name
        
        if feature_image_type == FeatureImageTypes.ATLAS_COORD:
            self.feature_names += [f"ATLAS_COORD_{d}" for d in ["x", "y", "z"]]
        else:
            self.feature_names.append(feature_name)

        feature_path = os.path.join(self.img.path, f"{feature_name}.nii.gz")

        if not os.path.exists(feature_path):
            print(f"For id {self.img.id_} feature {feature_name} is not computed!")
            return

        self.img.feature_images[feature_image_type] = sitk.ReadImage(feature_path)

    def _save_features(self):
        feature_names = []
        for feature_image_type, feature_image in self.img.feature_images.items():
            if feature_image_type in [FeatureImageTypes.T1w_INTENSITY, FeatureImageTypes.T2w_INTENSITY]:
                continue

            feature_name = feature_image_type.name
            feature_names.append(feature_name)

            feature_path = os.path.join(self.img.path, f"{feature_name}.nii.gz")

            if os.path.exists(feature_path) and not self.overwrite:
                print(f"For id {self.img.id_} feature {feature_name} has been already computed.")
                continue

            sitk.WriteImage(feature_image, feature_path)

        print(f"For id {self.img.id_} features saved: ({feature_names})")

    def _compute_features(self):
        if self.coordinates_feature:
            feature_image_type = FeatureImageTypes.ATLAS_COORD
            self.feature_names += [f"ATLAS_COORD_{d}" for d in ["x", "y", "z"]]
            
            atlas_coordinates = fltr_feat.AtlasCoordinates()
            self.img.feature_images[feature_image_type] = \
                atlas_coordinates.execute(self.img.images[structure.BrainImageTypes.T1w])

        if self.gradient_intensity_feature:
            feature_image_type = FeatureImageTypes.T1w_GRADIENT_INTENSITY
            self.feature_names.append(feature_image_type.name)
            self.img.feature_images[feature_image_type] = \
                sitk.GradientMagnitude(self.img.images[structure.BrainImageTypes.T1w])
            if self.t2_features:
                feature_image_type = FeatureImageTypes.T2w_GRADIENT_INTENSITY
                self.feature_names.append(feature_image_type.name)
                self.img.feature_images[feature_image_type] = \
                    sitk.GradientMagnitude(self.img.images[structure.BrainImageTypes.T2w])

        texture_features = []
        if self.texture_contrast_feature:
            texture_features.append("contrast")

        if self.texture_dissimilarity_feature:
            texture_features.append("dissimilarity")

        if self.texture_correlation_feature:
            texture_features.append("correlation")

        if len(texture_features):
            texture_features_extractor = fltr_feat.TextureFeatureExtractor()
            texture_features_images_t1 = texture_features_extractor.execute(
                self.img.images[structure.BrainImageTypes.T1w],
                texture_features
            )
            if self.t2_features:
                texture_features_images_t2 = texture_features_extractor.execute(
                    self.img.images[structure.BrainImageTypes.T2w],
                    texture_features
                )

            if self.texture_contrast_feature:
                feature_image_type = FeatureImageTypes.T1w_TEXTURE_CONTRAST
                self.feature_names.append(feature_image_type.name)
                self.img.feature_images[feature_image_type] = texture_features_images_t1["contrast"]

                if self.t2_features:
                    feature_image_type = FeatureImageTypes.T2w_TEXTURE_CONTRAST
                    self.feature_names.append(feature_image_type.name)
                    self.img.feature_images[feature_image_type] = texture_features_images_t2["contrast"]

            if self.texture_dissimilarity_feature:
                feature_image_type = FeatureImageTypes.T1w_TEXTURE_DISSIMILARITY
                self.feature_names.append(feature_image_type.name)
                self.img.feature_images[feature_image_type] = texture_features_images_t1["dissimilarity"]
                if self.t2_features:
                    feature_image_type = FeatureImageTypes.T2w_TEXTURE_DISSIMILARITY
                    self.feature_names.append(feature_image_type.name)
                    self.img.feature_images[feature_image_type] = texture_features_images_t2["dissimilarity"]

            if self.texture_correlation_feature:
                feature_image_type = FeatureImageTypes.T1w_TEXTURE_CORRELATION
                self.feature_names.append(feature_image_type.name)
                self.img.feature_images[feature_image_type] = texture_features_images_t1["correlation"]
                if self.t2_features:
                    feature_image_type = FeatureImageTypes.T2w_TEXTURE_CORRELATION
                    self.feature_names.append(feature_image_type.name)
                    self.img.feature_images[feature_image_type] = texture_features_images_t2["correlation"]

        if self.edge_feature:
            feature_image_type = FeatureImageTypes.T1w_EDGES
            self.feature_names.append(feature_image_type.name)

            edge_features_extractor = fltr_feat.EdgesFeatureExtractor()
            self.img.feature_images[feature_image_type] = \
                edge_features_extractor.execute(self.img.images[structure.BrainImageTypes.T1w])
            if self.t2_features:
                feature_image_type = FeatureImageTypes.T2w_EDGES
                self.feature_names.append(feature_image_type.name)

                self.img.feature_images[feature_image_type] = \
                    edge_features_extractor.execute(self.img.images[structure.BrainImageTypes.T2w])

    def _generate_feature_matrix(self):
        """Generates a feature matrix."""

        mask = None
        if self.training:
            # generate a randomized mask where 1 represents voxels used for training
            # the mask needs to be binary, where the value 1 is considered as a voxel which is to be loaded
            # we have following labels:
            # - 0 (background)
            # - 1 (white matter)
            # - 2 (grey matter)
            # - 3 (Hippocampus)
            # - 4 (Amygdala)
            # - 5 (Thalamus)

            # you can exclude background voxels from the training mask generation
            # mask_background = self.img.images[structure.BrainImageTypes.BrainMask]
            # and use background_mask=mask_background in get_mask()

            mask = fltr_feat.RandomizedTrainingMaskGenerator.get_mask(
                self.img.images[structure.BrainImageTypes.GroundTruth],
                [0, 1, 2, 3, 4, 5],
                [0.0003, 0.004, 0.003, 0.04, 0.04, 0.02])

            # convert the mask to a logical array where value 1 is False and value 0 is True
            mask = sitk.GetArrayFromImage(mask)
            mask = np.logical_not(mask)

        # generate features
        data = np.concatenate(
            [self._image_as_numpy_array(image, mask) for id_, image in self.img.feature_images.items()],
            axis=1)

        # generate labels (note that we assume to have a ground truth even for testing)
        labels = self._image_as_numpy_array(self.img.images[structure.BrainImageTypes.GroundTruth], mask)

        self.img.feature_matrix = (data.astype(np.float32), labels.astype(np.int16))

    @staticmethod
    def _image_as_numpy_array(image: sitk.Image, mask: np.ndarray = None):
        """Gets an image as numpy array where each row is a voxel and each column is a feature.

        Args:
            image (sitk.Image): The image.
            mask (np.ndarray): A mask defining which voxels to return. True is background, False is a masked voxel.

        Returns:
            np.ndarray: An array where each row is a voxel and each column is a feature.
        """

        number_of_components = image.GetNumberOfComponentsPerPixel()  # the number of features for this image
        no_voxels = np.prod(image.GetSize())
        image = sitk.GetArrayFromImage(image)

        if mask is not None:
            no_voxels = np.size(mask) - np.count_nonzero(mask)

            if number_of_components == 1:
                masked_image = np.ma.masked_array(image, mask=mask)
            else:
                # image is a vector image, make a vector mask
                vector_mask = np.expand_dims(mask, axis=3)  # shape is now (z, x, y, 1)
                vector_mask = np.repeat(vector_mask, number_of_components,
                                        axis=3)  # shape is now (z, x, y, number_of_components)
                masked_image = np.ma.masked_array(image, mask=vector_mask)

            image = masked_image[~masked_image.mask]

        return image.reshape((no_voxels, number_of_components))


def pre_process(id_: str, paths: dict, **kwargs) -> structure.BrainImage:
    """Loads and processes an image.

    The processing includes:

    - Registration
    - Pre-processing
    - Feature extraction

    Args:
        id_ (str): An image identifier.
        paths (dict): A dict, where the keys are an image identifier of type structure.BrainImageTypes
            and the values are paths to the images.

    Returns:
        (structure.BrainImage):
    """

    print('-' * 10, 'Processing', id_)

    # load image
    path = paths.pop(id_, '')  # the value with key id_ is the root directory of the image
    path_to_transform = paths.pop(structure.BrainImageTypes.RegistrationTransform, '')
    img = {img_key: sitk.ReadImage(path) for img_key, path in paths.items()}
    transform = sitk.ReadTransform(path_to_transform)
    img = structure.BrainImage(id_, path, img, transform)

    if not kwargs.get('load_pre', False):

        # construct pipeline for brain mask registration
        pipeline_brain_mask = fltr.FilterPipeline()
        if kwargs.get('registration_pre', False):
            pipeline_brain_mask.add_filter(fltr_prep.ImageRegistration())
            pipeline_brain_mask.set_param(fltr_prep.ImageRegistrationParameters(atlas_t1, img.transformation, True),
                                          len(pipeline_brain_mask.filters) - 1)

        # execute pipeline on the brain mask image
        img.images[structure.BrainImageTypes.BrainMask] = pipeline_brain_mask.execute(
            img.images[structure.BrainImageTypes.BrainMask])

        # construct pipeline for T1w image pre-processing
        pipeline_t1 = fltr.FilterPipeline()
        if kwargs.get('registration_pre', False):
            pipeline_t1.add_filter(fltr_prep.ImageRegistration())
            pipeline_t1.set_param(fltr_prep.ImageRegistrationParameters(atlas_t1, img.transformation),
                                  len(pipeline_t1.filters) - 1)
        if kwargs.get('skullstrip_pre', False):
            pipeline_t1.add_filter(fltr_prep.SkullStripping())
            pipeline_t1.set_param(fltr_prep.SkullStrippingParameters(img.images[structure.BrainImageTypes.BrainMask]),
                                  len(pipeline_t1.filters) - 1)
        if kwargs.get('normalization_pre', False):
            pipeline_t1.add_filter(fltr_prep.ImageNormalization())

        # execute pipeline on the T1w image
        img.images[structure.BrainImageTypes.T1w] = pipeline_t1.execute(img.images[structure.BrainImageTypes.T1w])

        # construct pipeline for T2w image pre-processing
        pipeline_t2 = fltr.FilterPipeline()
        if kwargs.get('registration_pre', False):
            pipeline_t2.add_filter(fltr_prep.ImageRegistration())
            pipeline_t2.set_param(fltr_prep.ImageRegistrationParameters(atlas_t2, img.transformation),
                                  len(pipeline_t2.filters) - 1)
        if kwargs.get('skullstrip_pre', False):
            pipeline_t2.add_filter(fltr_prep.SkullStripping())
            pipeline_t2.set_param(fltr_prep.SkullStrippingParameters(img.images[structure.BrainImageTypes.BrainMask]),
                                  len(pipeline_t2.filters) - 1)
        if kwargs.get('normalization_pre', False):
            pipeline_t2.add_filter(fltr_prep.ImageNormalization())

        # execute pipeline on the T2w image
        img.images[structure.BrainImageTypes.T2w] = pipeline_t2.execute(img.images[structure.BrainImageTypes.T2w])

        # construct pipeline for ground truth image pre-processing
        pipeline_gt = fltr.FilterPipeline()
        if kwargs.get('registration_pre', False):
            pipeline_gt.add_filter(fltr_prep.ImageRegistration())
            pipeline_gt.set_param(fltr_prep.ImageRegistrationParameters(atlas_t1, img.transformation, True),
                                  len(pipeline_gt.filters) - 1)

        # execute pipeline on the ground truth image
        img.images[structure.BrainImageTypes.GroundTruth] = pipeline_gt.execute(
            img.images[structure.BrainImageTypes.GroundTruth])

    if kwargs.get('registration_pre', False) and kwargs.get('save_pre', False):
        for img_key, path in paths.items():
            file_dir = os.path.dirname(path)
            pre_file_name = f"pre_{os.path.basename(path)}"
            pre_path = os.path.join(file_dir, pre_file_name)
            sitk.WriteImage(img.images[img_key], pre_path)

    else:
        print(f"Preprocessed images loaded from {path}")

    # update image properties to atlas image properties after registration
    img.image_properties = conversion.ImageProperties(img.images[structure.BrainImageTypes.T1w])

    # extract the features
    feature_extractor = FeatureExtractor(img, **kwargs)
    img = feature_extractor.execute()

    img.feature_images = {}  # we free up memory because we only need the img.feature_matrix
    # for training of the classifier

    return img


def post_process(img: structure.BrainImage, segmentation: sitk.Image, probability: sitk.Image,
                 **kwargs) -> sitk.Image:
    """Post-processes a segmentation.

    Args:
        img (structure.BrainImage): The image.
        segmentation (sitk.Image): The segmentation (label image).
        probability (sitk.Image): The probabilities images (a vector image).

    Returns:
        sitk.Image: The post-processed image.
    """

    print('-' * 10, 'Post-processing', img.id_)

    # construct pipeline
    pipeline = fltr.FilterPipeline()
    if kwargs.get('simple_post', False):
        pipeline.add_filter(fltr_postp.ImagePostProcessing())
    if kwargs.get('crf_post', False):
        pipeline.add_filter(fltr_postp.DenseCRF())
        pipeline.set_param(fltr_postp.DenseCRFParams(img.images[structure.BrainImageTypes.T1w],
                                                     img.images[structure.BrainImageTypes.T2w],
                                                     probability), len(pipeline.filters) - 1)

    return pipeline.execute(segmentation)


def init_evaluator() -> eval_.Evaluator:
    """Initializes an evaluator.

    Returns:
        eval.Evaluator: An evaluator.
    """

    # initialize metrics
    metrics = [metric.DiceCoefficient(), metric.HausdorffDistance(percentile=95)]
    # todo: add hausdorff distance, 95th percentile (see metric.HausdorffDistance)
    warnings.warn('Initialized evaluation with the Dice coefficient. Do you know other suitable metrics?')

    # define the labels to evaluate
    labels = {1: 'WhiteMatter',
              2: 'GreyMatter',
              3: 'Hippocampus',
              4: 'Amygdala',
              5: 'Thalamus'
              }

    evaluator = eval_.SegmentationEvaluator(metrics, labels)
    return evaluator


def pre_process_batch(data_batch: t.Dict[structure.BrainImageTypes, structure.BrainImage],
                      pre_process_params: dict = None, multi_process: bool = True) -> t.List[structure.BrainImage]:
    """Loads and pre-processes a batch of images.

    The pre-processing includes:

    - Registration
    - Pre-processing
    - Feature extraction

    Args:
        data_batch (Dict[structure.BrainImageTypes, structure.BrainImage]): Batch of images to be processed.
        pre_process_params (dict): Pre-processing parameters.
        multi_process (bool): Whether to use the parallel processing on multiple cores or to run sequentially.

    Returns:
        List[structure.BrainImage]: A list of images.
    """
    if pre_process_params is None:
        pre_process_params = {}

    params_list = list(data_batch.items())
    if multi_process:
        images = mproc.MultiProcessor.run(pre_process, params_list, pre_process_params, mproc.PreProcessingPickleHelper)
    else:
        images = [pre_process(id_, path, **pre_process_params) for id_, path in params_list]
    return images


def post_process_batch(brain_images: t.List[structure.BrainImage], segmentations: t.List[sitk.Image],
                       probabilities: t.List[sitk.Image], post_process_params: dict = None,
                       multi_process: bool = True) -> t.List[sitk.Image]:
    """ Post-processes a batch of images.

    Args:
        brain_images (List[structure.BrainImageTypes]): Original images that were used for the prediction.
        segmentations (List[sitk.Image]): The predicted segmentation.
        probabilities (List[sitk.Image]): The prediction probabilities.
        post_process_params (dict): Post-processing parameters.
        multi_process (bool): Whether to use the parallel processing on multiple cores or to run sequentially.

    Returns:
        List[sitk.Image]: List of post-processed images
    """
    if post_process_params is None:
        post_process_params = {}

    param_list = zip(brain_images, segmentations, probabilities)
    if multi_process:
        pp_images = mproc.MultiProcessor.run(post_process, param_list, post_process_params,
                                             mproc.PostProcessingPickleHelper)
    else:
        pp_images = [post_process(img, seg, prob, **post_process_params) for img, seg, prob in param_list]
    return pp_images
