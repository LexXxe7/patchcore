import os
import csv
import logging
import random
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch

LOGGER = logging.getLogger(__name__)


def plot_segmentation_images(
    savefolder,
    image_paths,
    segmentations,
    optimal_threshold,
    anomaly_scores=None,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth=4,
):
    """
    Generates anomaly segmentation images.

    Args:
        savefolder: [str]. Path to save folder.
        image_paths: [list of str]. List of paths to images.
        segmentations: [list of np.ndarray]. Generated anomaly segmentations.
        optimal_threshold: [float]. Pixel-wise optimal threshold.
        anomaly_scores: [list of float]. Anomaly scores for each image.
        mask_paths: [list of str]. List of paths to ground truth masks.
        image_transform: [function or lambda]. Optional transformation of images.
        mask_transform: [function or lambda]. Optional transformation of masks.
        save_depth: [int]. Number of path-strings to use for image savenames.
    """
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"

    os.makedirs(savefolder, exist_ok=True)

    for image_path, mask_path, anomaly_score, segmentation in tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        image = Image.open(image_path).convert("RGB")
        image = image_transform(image)
        if not isinstance(image, np.ndarray):
            image = image.numpy()

        if masks_provided:
            if mask_path is not None:
                mask = Image.open(mask_path).convert("RGB")
                mask = mask_transform(mask)
                if not isinstance(mask, np.ndarray):
                    mask = mask.numpy()
            else:
                mask = np.zeros_like(image)

        segmentation_mask = segmentation >= optimal_threshold
        segmentation = cv2.applyColorMap(np.uint8(255 * segmentation), cv2.COLORMAP_HOT)

        alpha = 0.5
        image_with_segmentation = cv2.addWeighted(
            image.transpose(1, 2, 0), alpha, segmentation, 1.0 - alpha, 0.0
        )

        contours, hierarchy = cv2.findContours(
            segmentation_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        fat_contours_image_color = cv2.drawContours(
            np.zeros_like(image_with_segmentation), contours, -1, (255, 255, 255), 2
        )
        fat_contours_image_gray = cv2.cvtColor(
            fat_contours_image_color, cv2.COLOR_BGR2GRAY
        )
        fat_contours = np.argwhere(fat_contours_image_gray == 255)
        for i, j in fat_contours:
            image_with_segmentation[i, j] = 0

        fat_contours_image = cv2.drawContours(
            np.zeros_like(image_with_segmentation), contours, -1, (100, 100, 100), 2
        )
        fat_contours_segmentation = cv2.applyColorMap(
            fat_contours_image, cv2.COLORMAP_HOT
        )
        image_with_segmentation = cv2.addWeighted(
            image_with_segmentation, 1.0, fat_contours_segmentation, 1.0, 0.0
        )

        savename = image_path.split("/")
        savename = "_".join(savename[-save_depth:])
        savefoldername = savename.split("\\")
        savefoldername = savefoldername[:-1]
        savefoldername = "\\".join(savefoldername)
        savefoldername = os.path.join(savefolder, savefoldername)
        if not os.path.isdir(savefoldername):
            os.makedirs(savefoldername, exist_ok=True)
        savename = os.path.join(savefolder, savename)

        f, axes = plt.subplots(2, 2)
        axes[0, 0].imshow(image.transpose(1, 2, 0))
        axes[0, 1].imshow(mask.transpose(1, 2, 0))
        axes[1, 0].imshow(cv2.cvtColor(segmentation, cv2.COLOR_BGR2RGB))
        axes[1, 1].imshow(cv2.cvtColor(image_with_segmentation, cv2.COLOR_BGR2RGB))
        f.set_size_inches(9, 9)
        f.tight_layout()
        f.savefig(savename)
        plt.close()


def create_storage_folder(
    main_folder_path, project_folder, group_folder, mode="iterate"
):
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder)
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path, group_folder + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    return save_path


def set_torch_device(gpu_ids):
    """
    Returns correct torch.device.

    Args:
        gpu_ids: [list]. List of GPU ids. If empty, CPU is used.
    """
    if len(gpu_ids):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device(f"cuda:{gpu_ids[0]}")
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """
    Fixed available seeds for reproducibility.

    Args:
        seed: [int]. Seed value.
        with_torch: [bool]. Flag. If true, torch-related seeds are fixed.
        with_cuda: [bool]. Flag. If true, torch+cuda-related seeds are fixed.
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
    column_names=[
        "Instance AUROC",
        "Full Pixel AUROC",
        "Full PRO",
        "Anomaly Pixel AUROC",
        "Anomaly PRO",
    ],
):
    """
    Stores computed results as CSV file.

    Args:
        results_path: [str]. Where to store result csv.
        results: [list of list]. List of lists containing results per dataset,
                 with results[i][0] == "dataset_name" and results[i][1:6] =
                 [instance_auroc, full_pixelwise_auroc, full_pro,
                 anomaly-only_pixelwise_auroc, anomaly-only_pro].
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
        LOGGER.info(f"{result_key}: {mean_metrics[result_key]:3.3f}")

    savename = os.path.join(results_path, "results.csv")
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {f"mean_{key}": value for key, value in mean_metrics.items()}
    return mean_metrics
