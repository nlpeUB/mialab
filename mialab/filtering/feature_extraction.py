"""The feature extraction module contains classes for feature extraction."""
import sys

import numpy as np
import pymia.filtering.filter as fltr
import SimpleITK as sitk
from typing import Tuple, Dict, List
from scipy.ndimage import sobel, laplace
from skimage.feature import canny
from radiomics import glcm
from skimage.feature import graycomatrix, graycoprops


class AtlasCoordinates(fltr.Filter):
    """Represents an atlas coordinates feature extractor."""

    def __init__(self):
        """Initializes a new instance of the AtlasCoordinates class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: fltr.FilterParams = None) -> sitk.Image:
        """Executes a atlas coordinates feature extractor on an image.

        Args:
            image (sitk.Image): The image.
            params (fltr.FilterParams): The parameters (unused).

        Returns:
            sitk.Image: The atlas coordinates image
            (a vector image with 3 components, which represent the physical x, y, z coordinates in mm).

        Raises:
            ValueError: If image is not 3-D.
        """

        if image.GetDimension() != 3:
            raise ValueError('image needs to be 3-D')

        x, y, z = image.GetSize()

        # create matrix with homogenous indices in axis 3
        coords = np.zeros((x, y, z, 4))
        coords[..., 0] = np.arange(x)[:, np.newaxis, np.newaxis]
        coords[..., 1] = np.arange(y)[np.newaxis, :, np.newaxis]
        coords[..., 2] = np.arange(z)[np.newaxis, np.newaxis, :]
        coords[..., 3] = 1

        # reshape such that each voxel is one row
        lin_coords = np.reshape(coords, [coords.shape[0] * coords.shape[1] * coords.shape[2], 4])

        # generate transformation matrix
        tmp_mat = image.GetDirection() + image.GetOrigin()
        tfm = np.reshape(tmp_mat, [3, 4], order='F')
        tfm = np.vstack((tfm, [0, 0, 0, 1]))

        atlas_coords = (tfm @ np.transpose(lin_coords))[0:3, :]
        atlas_coords = np.reshape(np.transpose(atlas_coords), [z, y, x, 3], 'F')

        img_out = sitk.GetImageFromArray(atlas_coords)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'AtlasCoordinates:\n' \
            .format(self=self)


def first_order_texture_features_function(values):
    """Calculates first-order texture features.

    Args:
        values (np.array): The values to calculate the first-order texture features from.

    Returns:
        np.array: A vector containing the first-order texture features:

            - mean
            - variance
            - sigma
            - skewness
            - kurtosis
            - entropy
            - energy
            - snr
            - min
            - max
            - range
            - percentile10th
            - percentile25th
            - percentile50th
            - percentile75th
            - percentile90th
    """
    eps = sys.float_info.epsilon  # to avoid division by zero

    mean = np.mean(values)
    std = np.std(values)
    snr = mean / std if std != 0 else 0
    min_ = np.min(values)
    max_ = np.max(values)
    num_values = len(values)
    p = values / (np.sum(values) + eps)
    return np.array([mean,
                     np.var(values),  # variance
                     std,
                     np.sqrt(num_values * (num_values - 1)) / (num_values - 2) * np.sum((values - mean) ** 3) /
                     (num_values * std ** 3 + eps),  # adjusted Fisher-Pearson coefficient of skewness
                     np.sum((values - mean) ** 4) / (num_values * std ** 4 + eps),  # kurtosis
                     np.sum(-p * np.log2(p)),  # entropy
                     np.sum(p ** 2),  # energy (intensity histogram uniformity)
                     snr,
                     min_,
                     max_,
                     max_ - min_,
                     np.percentile(values, 10),
                     np.percentile(values, 25),
                     np.percentile(values, 50),
                     np.percentile(values, 75),
                     np.percentile(values, 90)
                     ])


class NeighborhoodFeatureExtractor(fltr.Filter):
    """Represents a feature extractor filter, which works on a neighborhood."""

    def __init__(self, kernel=(3, 3, 3), function_=first_order_texture_features_function, neighborhood_radius: int = 3):
        """Initializes a new instance of the NeighborhoodFeatureExtractor class."""
        super().__init__()
        self.neighborhood_radius = neighborhood_radius
        self.kernel = kernel
        self.function = function_

    def execute(self, image: sitk.Image, params: fltr.FilterParams = None) -> sitk.Image:
        """Executes a neighborhood feature extractor on an image.

        Args:
            image (sitk.Image): The image.
            params (fltr.FilterParams): The parameters (unused).

        Returns:
            sitk.Image: The normalized image.

        Raises:
            ValueError: If image is not 3-D.
        """

        if image.GetDimension() != 3:
            raise ValueError('image needs to be 3-D')

        # test the function and get the output dimension for later reshaping
        function_output = self.function(np.array([1, 2, 3]))
        if np.isscalar(function_output):
            img_out = sitk.Image(image.GetSize(), sitk.sitkFloat32)
        elif not isinstance(function_output, np.ndarray):
            raise ValueError('function must return a scalar or a 1-D np.ndarray')
        elif function_output.ndim > 1:
            raise ValueError('function must return a scalar or a 1-D np.ndarray')
        elif function_output.shape[0] <= 1:
            raise ValueError('function must return a scalar or a 1-D np.ndarray with at least two elements')
        else:
            img_out = sitk.Image(image.GetSize(), sitk.sitkVectorFloat32, function_output.shape[0])

        img_out_arr = sitk.GetArrayFromImage(img_out)
        img_arr = sitk.GetArrayFromImage(image)
        z, y, x = img_arr.shape

        z_offset = self.kernel[2]
        y_offset = self.kernel[1]
        x_offset = self.kernel[0]
        pad = ((0, z_offset), (0, y_offset), (0, x_offset))
        img_arr_padded = np.pad(img_arr, pad, 'symmetric')

        for xx in range(0, x, self.neighborhood_radius):
            for yy in range(0, y, self.neighborhood_radius):
                for zz in range(0, z, self.neighborhood_radius):
                    val = self.function(img_arr_padded[zz:zz + z_offset, yy:yy + y_offset, xx:xx + x_offset])
                    img_out_arr[zz, yy, xx] = val

        img_out = sitk.GetImageFromArray(img_out_arr)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'NeighborhoodFeatureExtractor:\n' \
            .format(self=self)


class RandomizedTrainingMaskGenerator:
    """Represents a training mask generator.

    A training mask is an image with intensity values 0 and 1, where 1 represents masked.
    Such a mask can be used to sample voxels for training.
    """

    @staticmethod
    def get_mask(ground_truth: sitk.Image,
                 ground_truth_labels: list,
                 label_percentages: list,
                 background_mask: sitk.Image = None) -> sitk.Image:
        """Gets a training mask.

        Args:
            ground_truth (sitk.Image): The ground truth image.
            ground_truth_labels (list of int): The ground truth labels,
                where 0=background, 1=label1, 2=label2, ..., e.g. [0, 1]
            label_percentages (list of float): The percentage of voxels of a corresponding label to extract as mask,
                e.g. [0.2, 0.2].
            background_mask (sitk.Image): A mask, where intensity 0 indicates voxels to exclude independent of the
            label.

        Returns:
            sitk.Image: The training mask.
        """

        # initialize mask
        ground_truth_array = sitk.GetArrayFromImage(ground_truth)
        mask_array = np.zeros(ground_truth_array.shape, dtype=np.uint8)

        # exclude background
        if background_mask is not None:
            background_mask_array = sitk.GetArrayFromImage(background_mask)
            background_mask_array = np.logical_not(background_mask_array)
            ground_truth_array = ground_truth_array.astype(float)  # convert to float because of np.nan
            ground_truth_array[background_mask_array] = np.nan

        for label_idx, label in enumerate(ground_truth_labels):
            indices = np.transpose(np.where(ground_truth_array == label))
            np.random.shuffle(indices)

            no_mask_items = int(indices.shape[0] * label_percentages[label_idx])

            for no in range(no_mask_items):
                x = indices[no][0]
                y = indices[no][1]
                z = indices[no][2]
                mask_array[x, y, z] = 1  # this is a masked item

        mask = sitk.GetImageFromArray(mask_array)
        mask.SetOrigin(ground_truth.GetOrigin())
        mask.SetDirection(ground_truth.GetDirection())
        mask.SetSpacing(ground_truth.GetSpacing())

        return mask


class PyRadiomicsTextureFeatureExtractor(fltr.Filter):
    """Represents a feature extractor filter, which extracts texture features."""

    def execute(self, image: sitk.Image, brain_mask: np.array, params: fltr.FilterParams = None) -> sitk.Image:
        """Executes a texture feature extractor on an image with PyRadiomics library.

        Args:
            image (sitk.Image): The image.
            brain_mask (np.array): The mask of the brain.
            params (fltr.FilterParams): The parameters (unused).

        Returns:
            sitk.Image: The texture features in image format.

        Raises:
            ValueError: If image is not 3-D.
        """

        if image.GetDimension() != 3:
            raise ValueError('image needs to be 3-D')

        glcm_features = glcm.RadiomicsGLCM(image, brain_mask)
        glcm_features.disableAllFeatures()

        glcm_features_names = [
            'Autocorrelation',
            'Contrast',
            'InverseVariance'
        ]  # selected from glcm_features.getFeatureNames()

        for name in glcm_features_names:
            glcm_features.enableFeatureByName(name)

        results = glcm_features.execute()

        return results


class TextureFeatureExtractor(fltr.Filter):
    """Represents a feature extractor filter, which extracts texture features."""

    def execute(self, image: sitk.Image, features: List[str], params: fltr.FilterParams = None) -> Dict:
        """Executes a texture feature extractor on an image.
        Args:
            image (sitk.Image): The image.
            features (List[str]): List of features to extract from patches with GLCM.
            params (fltr.FilterParams): The parameters (unused).
        Returns:
            Dict[str: sitk.Image]: The dict with texture features in image format.
        Raises:
            ValueError: If image is not 3-D.
        """

        if image.GetDimension() != 3:
            raise ValueError('image needs to be 3-D')

        img_arr = sitk.GetArrayFromImage(image)
        glcm_features = self._compute_glcm_features_per_patch(img_arr, features, patch_size=(5, 5))

        img_outs = {}
        for feature in features:
            glcm_feature = glcm_features[feature]
            img_out = sitk.GetImageFromArray(glcm_feature)
            img_out.CopyInformation(image)
            img_outs[feature] = img_out

        return img_outs

    @staticmethod
    # TODO: Adapt the patch size and distance
    def _compute_glcm_features_per_patch(image_3d, features, patch_size=(10, 10), step=10, distance=3):
        image_3d = image_3d / np.max(image_3d)
        image_3d = (255 * image_3d).astype(int)

        output_images = {}
        for feature in features:
            output_image = np.zeros_like(image_3d, dtype=float)
            depth, height, width = image_3d.shape

            for z in range(depth):
                slice_2d = image_3d[z]
                output_slice = np.zeros_like(slice_2d, dtype=float)

                for i in range(0, height, step):
                    for j in range(0, width, step):
                        if i + patch_size[0] <= height and j + patch_size[1] <= width:
                            patch = slice_2d[i:i + patch_size[0], j:j + patch_size[1]]

                            glcm = graycomatrix(patch, distances=[distance], angles=[0], levels=256, symmetric=True, normed=True)
                            feature_from_glcm = graycoprops(glcm, feature)[0, 0]

                            output_slice[i:i + patch_size[0], j:j + patch_size[1]] = feature_from_glcm

                output_image[z] = output_slice
            output_images[feature] = output_image

        return output_images


class EdgesFeatureExtractor(fltr.Filter):
    """Represents a feature extractor filter, which extracts edges."""

    def execute(self, image: sitk.Image, params: fltr.FilterParams = None) -> sitk.Image:
        """Executes an edges extractor on an image.

        Args:
            image (sitk.Image): The image.
            params (fltr.FilterParams): The parameters (unused).

        Returns:
            sitk.Image: The edges features in image format.

        Raises:
            ValueError: If image is not 3-D.
        """

        if image.GetDimension() != 3:
            raise ValueError('image needs to be 3-D')

        img_arr = sitk.GetArrayFromImage(image)

        # TODO: In case we really need it, we can turn on different edges detectors
        sobel_edges, _, _ = self._calculate_edge_features(img_arr)
        edges_features = sobel_edges

        img_out = sitk.GetImageFromArray(edges_features)
        img_out.CopyInformation(image)

        return img_out

    @staticmethod
    def _calculate_edge_features(image: sitk.Image) -> Tuple:
        sobel_edges = np.sqrt(sobel(image, axis=0) ** 2 + sobel(image, axis=1) ** 2 + sobel(image, axis=2) ** 2)
        laplacian_edges = laplace(image)

        canny_edges = []
        for slice_idx in range(image.shape[2]):
            slice_2d = image[:, :, slice_idx]
            edges = canny(slice_2d)
            canny_edges.append(edges)
        canny_edges = np.stack(canny_edges, axis=2)

        return sobel_edges, laplacian_edges, canny_edges
