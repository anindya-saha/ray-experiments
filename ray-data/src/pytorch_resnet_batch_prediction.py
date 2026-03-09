import ray
import ray.data

import torch
import torchvision
from torchvision import transforms

import numpy as np
from typing import Dict, List, Union

import tempfile

import logging


logger = logging.getLogger(__name__)

class PreprocessImage:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            torchvision.models.ResNet152_Weights.IMAGENET1K_V1.transforms(),
        ])

    def __call__(self, row: Dict[str, np.ndarray]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        image = np.array(row["image"], copy=True)
        return {
            "original_image": row["image"],
            "transformed_image": self.transform(image),
        }


class ResNetPredictor:
    def __init__(self):
        self.weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchvision.models.resnet152(weights=self.weights).to(self.device)
        self.model.eval()


    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, Union[np.ndarray, List[str]]]:
        # Convert the numpy array of images into a PyTorch tensor.
        # Move the tensor batch to GPU if available.
        torch_batch = torch.from_numpy(batch["transformed_image"]).to(self.device)
        with torch.inference_mode():
            predictions = self.model(torch_batch)
            predicted_classes = predictions.argmax(dim=1).detach().cpu()
            predicted_labels = [
                self.weights.meta["categories"][class_index]for class_index in predicted_classes
            ]
            return {
                "original_image": batch["original_image"],
                "predicted_label": predicted_labels,
            }


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        ray.init()

        logger.info("Ray cluster resources: %s", ray.cluster_resources())
        logger.info("Ray available resources: %s", ray.available_resources())

        s3_uri = "s3://anonymous@air-example-data-2/imagenette2/train/"

        images_ds = ray.data.read_images(s3_uri, mode="RGB")
        
        logger.info("Dataset schema: %s", images_ds.schema())

        preprocessed_ds = images_ds.map(PreprocessImage, concurrency=32)  # pyright: ignore[reportArgumentType]
        
        predictions_ds = preprocessed_ds.map_batches(
            ResNetPredictor,  # pyright: ignore[reportArgumentType]
            num_gpus =1,    # Specify 1 GPU per model replica
            concurrency=4,  # Use 4 GPUs. Change this number based on the number of GPUs in your cluster.
            batch_size=8,   # Process 8 images at a time. Change this number based on the memory available on your GPUs.
        )
        logger.info("Predictions dataset schema: %s", predictions_ds.schema())

        temp_dir = tempfile.mkdtemp()
        predictions_ds.drop_columns(["original_image"]).write_parquet(f"local://{temp_dir}")
        logger.info("Predictions dataset written to %s", temp_dir)
    except Exception as e:
        logger.error("Error initializing Ray: %s", e)
        exit(1)
    finally:
        if ray.is_initialized():
            ray.shutdown()
