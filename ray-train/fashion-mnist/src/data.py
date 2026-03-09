import argparse
import logging
import os

from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Normalize, Compose

logger = logging.getLogger(__name__)

DEFAULT_TRANSFORM = Compose([ToTensor(), Normalize(0.5, 0.5)])


def download_dataset(data_dir):
    """Download FashionMNIST train and test splits if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    logger.info("Ensuring FashionMNIST is available at %s", data_dir)
    FashionMNIST(root=data_dir, train=True, download=True)
    FashionMNIST(root=data_dir, train=False, download=True)
    logger.info("FashionMNIST ready at %s", data_dir)


def load_dataset(data_dir, train=True, transform=None):
    """Load FashionMNIST from data_dir. Data must already be downloaded."""
    if transform is None:
        transform = DEFAULT_TRANSFORM
    return FashionMNIST(root=data_dir, train=train, download=False, transform=transform)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Download FashionMNIST dataset")
    parser.add_argument("--data-dir", type=str, required=True, help="Target directory")
    args = parser.parse_args()
    download_dataset(args.data_dir)
