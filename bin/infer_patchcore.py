import sys
import os
import contextlib
import gc
import logging
import click

import numpy as np
import torch
from torch.utils.data import DataLoader

import patchcore.common
import patchcore.utils
import patchcore.sampler
import patchcore.metrics
import patchcore.patchcore

LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"]}


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, multiple=True, default=[0], show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--save_segmentation_images", is_flag=True)
def main(**kwargs):
    pass


@main.result_callback()
def run(methods, results_path, gpu, seed, save_segmentation_images):
    methods = {key: value for key, value in methods}

    os.makedirs(results_path, exist_ok=True)

    device = patchcore.utils.set_torch_device(gpu)
    # Device context here is specifically set and used later because
    # there was GPU memory-bleeding which I could only fix with
    # context managers.
    device_context = (
        torch.cuda.device(f"cuda:{device.index}")
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []

    dataloader_iter, n_dataloaders = methods.get("get_dataloaders_iter")
    dataloader_iter = dataloader_iter(seed)
    patchcore_iter, n_patchcores = methods.get("get_patchcore_iter")
    patchcore_iter = patchcore_iter(device)
    if not (n_dataloaders == n_patchcores or n_patchcores == 1):
        raise ValueError(
            "Please ensure that #PatchCores == #Datasets or #PatchCores == 1!"
        )

    for dataloader_count, dataloaders in enumerate(dataloader_iter):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["testing"].name, dataloader_count + 1, n_dataloaders
            )
        )

        patchcore.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["testing"].name

        with device_context:
            torch.cuda.empty_cache()
            if dataloader_count < n_patchcores:
                PatchCore_list = next(patchcore_iter)

            aggregator = {"scores": [], "segmentations": []}
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    f"Embedding test data with models ({i + 1}/{len(PatchCore_list)})"
                )
                scores, segmentations, labels_gt, masks_gt = PatchCore.predict(
                    dataloaders["testing"]
                )
                aggregator.get("scores").append(scores)
                aggregator.get("segmentations").append(segmentations)

            scores = np.array(aggregator.get("scores"))
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = np.mean(scores, axis=0)

            segmentations = np.array(aggregator.get("segmentations"))
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]

            # (Optional) Plot segmentation images.
            if save_segmentation_images:
                image_paths = [
                    x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                mask_paths = [
                    x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                ]

                def image_transform(image):
                    in_mean = np.array(
                        dataloaders["testing"].dataset.transform_mean
                    ).reshape(-1, 1, 1)
                    in_std = np.array(
                        dataloaders["testing"].dataset.transform_std
                    ).reshape(-1, 1, 1)

                    image = dataloaders["testing"].dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)

                def mask_transform(mask):
                    return dataloaders["testing"].dataset.transform_mask(mask).numpy()

                image_save_path = os.path.join(
                    results_path, "segmentation_images", dataset_name
                )
                os.makedirs(image_save_path, exist_ok=True)
                patchcore.utils.plot_segmentation_images(
                    image_save_path,
                    image_paths,
                    segmentations,
                    anomaly_scores=scores,
                    mask_paths=mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )

            LOGGER.info("Computing evaluation metrics.")
            image_scores = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, anomaly_labels
            )
            auroc = image_scores.get("auroc")

            confusion_matrix_save_path = os.path.join(
                results_path, "confusion_matrices", dataset_name
            )
            os.makedirs(confusion_matrix_save_path, exist_ok=True)
            confusion_matrix_display = image_scores.get("confusion_matrix_display")
            confusion_matrix_display.plot().figure_.savefig(
                os.path.join(confusion_matrix_save_path, "confusion_matrix.png")
            )

            # Compute PRO score and PW Auroc for all images.
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt
            )
            full_pixel_auroc = pixel_scores.get("auroc")

            # Compute PRO score and PW Auroc only for images with anomalies.
            sel_idxs = []
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs.append(i)
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                [segmentations[i] for i in sel_idxs],
                [masks_gt[i] for i in sel_idxs],
            )
            anomaly_pixel_auroc = pixel_scores.get("auroc")

            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "confusion_matrix": confusion_matrix_display.confusion_matrix,
                    "instance_auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                }
            )

            for key, value in result_collect[-1].items():
                if key != "dataset_name":
                    if key == "confusion_matrix":
                        tn, fp, fn, tp = value.ravel()
                        LOGGER.info(f"{key}: {tn} {fp}\n" + " " * 33 + f"{fn} {tp}")
                    else:
                        LOGGER.info(f"{key}: {value:3.3f}")

            del PatchCore_list
            gc.collect()

        LOGGER.info("\n\n-----\n")

    # Store all results and mean scores to a CSV file.
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results.get("dataset_name") for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    patchcore.utils.compute_and_store_final_results(
        results_path,
        result_scores,
        row_names=result_dataset_names,
        column_names=result_metric_names,
    )


@main.command("patch_core_loader")
# Pretraining-specific parameters.
@click.option("--patch_core_paths", "-p", type=str, multiple=True, default=[])
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
def patch_core_loader(patch_core_paths, faiss_on_gpu, faiss_num_workers):
    def get_patchcore_iter(device):
        for patch_core_path in patch_core_paths:
            loaded_patchcores = []
            gc.collect()

            n_patchcores = len(
                [x for x in os.listdir(patch_core_path) if ".faiss" in x]
            )
            if n_patchcores == 1:
                nn_method = patchcore.common.FaissNN(
                    on_gpu=faiss_on_gpu,
                    num_workers=faiss_num_workers,
                )
                patchcore_instance = patchcore.patchcore.PatchCore(device)
                patchcore_instance.load_from_path(
                    load_path=patch_core_path, device=device, nn_method=nn_method
                )
                loaded_patchcores.append(patchcore_instance)
            else:
                for i in range(n_patchcores):
                    nn_method = patchcore.common.FaissNN(
                        on_gpu=faiss_on_gpu,
                        num_workers=faiss_num_workers,
                    )
                    patchcore_instance = patchcore.patchcore.PatchCore(device)
                    patchcore_instance.load_from_path(
                        load_path=patch_core_path,
                        device=device,
                        nn_method=nn_method,
                        prepend=f"Ensemble-{i + 1}-{n_patchcores}_",
                    )
                    loaded_patchcores.append(patchcore_instance)

            yield loaded_patchcores

    return ("get_patchcore_iter", [get_patchcore_iter, len(patch_core_paths)])


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", type=str, multiple=True, required=True)
@click.option("--batch_size", type=int, default=1, show_default=True)
@click.option("--resize", type=int, default=256, show_default=True)
@click.option("--imagesize", type=int, default=224, show_default=True)
@click.option("--num_workers", type=int, default=8, show_default=True)
@click.option("--augment", is_flag=True)
def dataset(
    name, data_path, subdatasets, batch_size, resize, imagesize, num_workers, augment
):
    dataset_info = _DATASETS.get(name)
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders_iter(seed):
        for subdataset in subdatasets:
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                source=data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            test_dataloader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader.name = name
            if subdataset is not None:
                test_dataloader.name += "_" + subdataset

            dataloader_dict = {"testing": test_dataloader}
            yield dataloader_dict

    return ("get_dataloaders_iter", [get_dataloaders_iter, len(subdatasets)])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info(f"Command line arguments: {' '.join(sys.argv)}")
    main()
