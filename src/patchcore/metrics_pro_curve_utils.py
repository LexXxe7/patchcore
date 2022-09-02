import numpy as np
from scipy.ndimage.measurements import label


class GroundTruthComponent:
    """
    Stores sorted anomaly scores of a single ground truth component.
    Used to efficiently compute the region overlap for many increasing
    thresholds.
    """

    def __init__(self, anomaly_scores):
        """
        Initializes the module.

        Args:
            anomaly_scores: List of all anomaly scores within the ground truth
                            component as np.ndarray.
        """
        # Keep a sorted list of all anomaly scores within the component.
        self.anomaly_scores = anomaly_scores.copy()
        self.anomaly_scores.sort()

        # Pointer to the anomaly score where the current threshold divides
        # the component into OK-/NOK-pixels.
        self.index = 0

        # The last evaluated threshold.
        self.last_threshold = None

    def compute_overlap(self, threshold):
        """
        Computes the region overlap for a specific threshold.
        Thresholds must be passed in increasing order.

        Args:
            threshold: Threshold to compute the region overlap.

        Returns:
            Region overlap for the specified threshold.
        """
        if self.last_threshold is not None:
            assert (
                self.last_threshold <= threshold
            ), "Last threshold must not be greater than specified threshold"

        # Increase the index until it points to an anomaly score that is just
        # above the specified threshold.
        while (
            self.index < len(self.anomaly_scores)
            and self.anomaly_scores[self.index] <= threshold
        ):
            self.index += 1

        # Compute the fraction of component pixels that are correctly segmented
        # as anomalous.
        return 1.0 - self.index / len(self.anomaly_scores)


def collect_anomaly_scores(anomaly_maps, ground_truth_maps):
    """
    Extracts the anomaly scores for each ground truth connected component
    as well as the anomaly scores for each potential false positive pixel from
    the anomaly maps.

    Args:
        anomaly_maps: List of anomaly maps (2D np.ndarrays) that contain a
                      real-valued anomaly score at each pixel.

        ground_truth_maps: List of ground truth maps (2D np.ndarrays) that
                           contain binary-valued ground truth labels for each
                           pixel.
                           0 indicates that a pixel is nominal.
                           1 indicates that a pixel is anomalous.

    Returns:
        ground_truth_components: A list of all ground truth connected components
                                 that appear in the dataset. For each component,
                                 a sorted list of its anomaly scores is stored.

        anomaly_scores_ok_pixels: A sorted list of the anomaly scores of all nominal
                                  pixels in the dataset. This list can be used to
                                  quickly select thresholds that fix a certain
                                  false positive rate.
    """
    # Make sure an anomaly map is present for each ground truth map.
    assert len(anomaly_maps) == len(
        ground_truth_maps
    ), "len(anomaly_maps) != len(ground_truth_maps)"

    # Initialize the ground truth components and scores of potential false positive
    # pixels.
    ground_truth_components = []
    anomaly_scores_ok_pixels = np.zeros(
        len(ground_truth_maps) * ground_truth_maps[0].size
    )

    # Structuring element for computing the connected components.
    structure = np.ones((3, 3), dtype=int)

    # Collect the anomaly scores within each ground truth component and for all
    # potential false positive pixels.
    ok_index = 0
    for gt_map, prediction in zip(ground_truth_maps, anomaly_maps):
        # Compute the connected components in the ground truth map.
        labeled, n_components = label(gt_map, structure)

        # Store all potential false positive pixels.
        num_ok_pixels = len(prediction[labeled == 0])
        anomaly_scores_ok_pixels[ok_index : ok_index + num_ok_pixels] = prediction[
            labeled == 0
        ].copy()
        ok_index += num_ok_pixels

        # Fetch the anomaly scores within each ground truth component.
        for k in range(n_components):
            component_scores = prediction[labeled == (k + 1)]
            ground_truth_components.append(GroundTruthComponent(component_scores))

    # Sort all potential false positive pixels.
    anomaly_scores_ok_pixels = np.resize(anomaly_scores_ok_pixels, ok_index)
    anomaly_scores_ok_pixels.sort()

    return ground_truth_components, anomaly_scores_ok_pixels


def compute_pro(anomaly_maps, ground_truth_maps, num_thresholds):
    """
    Computes the PRO curve at equidistant interpolation points for a set of
    anomaly maps with corresponding ground truth maps. The number of the
    interpolation points can be set manually.

    Args:
        anomaly_maps: List of anomaly maps (2D np.ndarrays) that contain a
                      real-valued anomaly score at each pixel.

        ground_truth_maps: List of ground truth maps (2D np.ndarrays) that
                           contain binary-valued ground truth labels for each
                           pixel.
                           0 indicates that a pixel is nominal.
                           1 indicates that a pixel is anomalous.

        num_thresholds: Number of thresholds to compute the PRO curve.

    Returns:
        fprs: List of the false positive rates.
        pros: List of the corresponding PRO values.
    """
    # Fetch the sorted anomaly scores.
    ground_truth_components, anomaly_scores_ok_pixels = collect_anomaly_scores(
        anomaly_maps, ground_truth_maps
    )

    # Select the equidistant thresholds.
    threshold_positions = np.linspace(
        0, len(anomaly_scores_ok_pixels) - 1, num=num_thresholds, dtype=int
    )

    fprs = [1.0]
    pros = [1.0]
    for pos in threshold_positions:
        threshold = anomaly_scores_ok_pixels[pos]

        # Compute the false positive rate for this threshold.
        fpr = 1.0 - (pos + 1) / len(anomaly_scores_ok_pixels)

        # Compute the PRO value for this threshold.
        pro = 0.0
        for component in ground_truth_components:
            pro += component.compute_overlap(threshold)
        pro /= len(ground_truth_components)

        fprs.append(fpr)
        pros.append(pro)

    # Return FPR-PRO-pairs in increasing FPR order.
    fprs = fprs[::-1]
    pros = pros[::-1]

    return fprs, pros
