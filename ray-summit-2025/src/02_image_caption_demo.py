"""
Image Captioning Demo using Ray Data, Ray Data LLM, and vLLM.

- Load dataset
- Dataset repartitioning
- Scale out Preprocess dataset
- Scale out Postprocess dataset

GPU-accelerated inference:
- Create LLM processor pipeline
- Run inference on the LLM processor pipeline
"""

from __future__ import annotations

import os
import gc
import time
import logging

from typing import Any
from pathlib import Path
from io import BytesIO

from PIL import Image

from dataclasses import dataclass, field

import datasets

import ray
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class JobConfig:
    """Configuration for image captioning job."""

    # Dataset configuration
    dataset_name: str = "imagenet-1k"
    dataset_split: str = "train[:10000]"

    # Processing configuration
    num_partitions: int = 32
    preprocessing_concurrency: int = 8
    preprocessing_num_cpus: int = 4
    postprocessing_concurrency: int = 8
    postprocessing_num_cpus: int = 4
    
    # Batch processing configuration
    preprocessing_batch_size: int = 32
    postprocessing_batch_size: int = 64

    # Model configuration
    model_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    max_model_len: int = 4096
    max_pixels: int = 1280 * 28 * 28
    min_pixels: int = 256 * 28 * 28

    # Inference configuration
    batch_size: int = 4             # Number of samples processed per batch (reduce if GPU memory is limited)
    num_inference_engines: int = 2  # Number of llm engines to run in parallel
    tensor_parallel_size: int = 1   # Number of GPUs per llm engine
    accelerator_type: str = "A10G"

    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 200

    # Output configuration
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    output_filename: str = "02_captioned_dataset.parquet"

    def __post_init__(self):
        """Ensure output directory exists."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_path(self) -> Path:
        """Full path to output file."""
        return self.output_dir / self.output_filename


class ImageCaptionPipelineV2:
    """Production-ready image captioning pipeline using Ray Data and vLLM."""

    def __init__(self, config: JobConfig):
        """Initialize the pipeline with configuration."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def caption_preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Batch preprocessing for image captioning - more efficient than single-row processing.
        
        Args:
            batch: Batch of rows containing image data
            
        Returns:
            Batch of preprocessed rows with messages and preserved image data
        """
        # Initialize output lists
        messages_list = []
        sampling_params_list = []
        image_bytes_list = []
        image_path_list = []
        
        # Handle different batch formats (pandas vs dict)
        if hasattr(batch, 'to_dict'):  # pandas DataFrame
            batch = batch.to_dict(orient='list')
        
        # Process each item in the batch
        images = batch["image"]
        for i, image_data in enumerate(images):
            # Handle different image schemas (cached vs HuggingFace)
            if isinstance(image_data, dict):
                # HuggingFace format with bytes
                pil_image = Image.open(BytesIO(image_data["bytes"]))
                image_path = image_data.get("path", "unknown")
                image_bytes = image_data["bytes"]
            else:
                # Direct PIL Image from cached dataset
                pil_image = image_data
                image_path = "unknown"
                # Convert PIL to bytes for preservation
                buffer = BytesIO()
                pil_image.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
            
            # Create messages for this image
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert image captioner. Provide detailed, "
                    "accurate descriptions of images.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please provide a detailed caption for this image. "
                            "Describe what you see, including objects, people, "
                            "actions, and the overall scene.",
                        },
                        {
                            "type": "image",
                            "image": pil_image,
                        },
                    ],
                },
            ]
            
            # Append to lists
            messages_list.append(messages)
            sampling_params_list.append({
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "detokenize": True,
            })
            image_bytes_list.append(image_bytes)
            image_path_list.append(image_path)
        
        return {
            "messages": messages_list,
            "sampling_params": sampling_params_list,
            "image_bytes": image_bytes_list,
            "image_path": image_path_list,
        }

    def caption_postprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Batch postprocessing - extract captions and preserve image data.
        
        Args:
            batch: Batch containing generated text and preserved data
            
        Returns:
            Batch of formatted rows with captions and image data
        """
        # Initialize output lists
        caption_list = []
        image_list = []
        
        # Handle different batch formats (pandas vs dict)
        if hasattr(batch, 'to_dict'):  # pandas DataFrame
            batch = batch.to_dict(orient='list')
        
        # Process each item in the batch
        for i in range(len(batch["generated_text"])):
            # Reconstruct image dict for visualization
            image_dict = {
                "bytes": batch["image_bytes"][i],
                "path": batch["image_path"][i],
            }
            
            caption_list.append(batch["generated_text"][i])
            image_list.append(image_dict)
        
        return {
            "caption": caption_list,
            "image": image_list,
        }

    def load_dataset(self) -> ray.data.Dataset:
        """Load and prepare the dataset."""
        self.logger.info(
            f"Loading dataset: {self.config.dataset_name} [{self.config.dataset_split}]"
        )

        hf_dataset = datasets.load_dataset(
            self.config.dataset_name, split=self.config.dataset_split
        )
        ray_dataset = ray.data.from_huggingface(hf_dataset)  # type: ignore[arg-type]

        self.logger.info(f"Dataset loaded: {ray_dataset.count()} samples")
        return ray_dataset

    def partition_dataset(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
        """Repartition the dataset into blocks."""
        self.logger.info(
            f"Repartitioning dataset into {self.config.num_partitions} blocks"
        )
        dataset = dataset.repartition(self.config.num_partitions)

        return dataset

    def preprocess_dataset(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
        """Apply preprocessing with parallelization using map_batches."""
        self.logger.info("Starting parallel batch preprocessing...")
        
        # Use map_batches for better efficiency
        preprocessed = dataset.map_batches(
            self.caption_preprocess_batch,
            batch_size=self.config.preprocessing_batch_size,  # Use configured batch size
            num_cpus=self.config.preprocessing_num_cpus,
            concurrency=self.config.preprocessing_concurrency,
            batch_format="pandas",  # Use pandas for better performance
        )

        return preprocessed

    def create_llm_processor(self):
        """Create and configure the vLLM processor."""
        processor_config = vLLMEngineProcessorConfig(
            model_source=self.config.model_name,
            engine_kwargs={
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "max_model_len": self.config.max_model_len,
                "enable_chunked_prefill": True,
                "enable_prefix_caching": True,
                "limit_mm_per_prompt": {"image": 1},
                "max_num_seqs": 64,
                "mm_processor_kwargs": {
                    "max_pixels": self.config.max_pixels,
                    "min_pixels": self.config.min_pixels,
                },
                "trust_remote_code": True,
                "gpu_memory_utilization": 0.9,  # Increase for better utilization
                #"disable_log_stats": False, # Critical: enable vLLM's internal logging. False by default.
                #"distributed_executor_backend": "ray",
            },
            runtime_env={
                "env_vars": {
                    "HF_TOKEN": os.environ["HF_TOKEN"],
                    "HF_HOME": os.environ["HF_HOME"],
                    "VLLM_USE_V1": "1",
                    #"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                },
            },
            batch_size=self.config.batch_size,
            accelerator_type=self.config.accelerator_type,
            concurrency=self.config.num_inference_engines,
            has_image=True,
        )

        return build_llm_processor(
            processor_config,
            preprocess=None,
            postprocess=None,
        )

    def postprocess_dataset(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
        """Apply postprocessing with parallelization using map_batches."""
        self.logger.info("Starting parallel batch postprocessing...")
        
        # Use map_batches for better efficiency
        postprocessed = dataset.map_batches(
            self.caption_postprocess_batch,
            batch_size=self.config.postprocessing_batch_size,  # Use configured batch size
            num_cpus=self.config.postprocessing_num_cpus,
            concurrency=self.config.postprocessing_concurrency,
            batch_format="pandas",  # Use pandas for better performance
        )

        return postprocessed

    def run(self) -> dict[str, Any]:
        """Execute the complete pipeline."""
        start_time = time.time()

        # Initialize Ray
        self.logger.info("Initializing Ray...")
        ray.init(
            _metrics_export_port=8080,
            runtime_env={
                "env_vars": {
                    "HF_TOKEN": os.environ["HF_TOKEN"],
                    "HF_HOME": os.environ["HF_HOME"],
                    #"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                }
            },
        )

        try:
            # Step 1: Load dataset
            ray_dataset = self.load_dataset()

            # Step 2: Repartition dataset
            partitioned_dataset = self.partition_dataset(ray_dataset)

            # Step 3: Preprocess dataset
            preprocessed_dataset = self.preprocess_dataset(partitioned_dataset)

            # Step 4: Create LLM processor
            llm_processor = self.create_llm_processor()

            # Step 5: Run inference
            self.logger.info("Starting inference...")
            inference_start = time.time()
            captioned_dataset = llm_processor(preprocessed_dataset)
            inference_time = time.time() - inference_start

            # Step 6: Postprocess
            postprocessed_dataset = self.postprocess_dataset(captioned_dataset)

            # Step 7: Save results
            self.logger.info(f"Saving results to: {self.config.output_path}")

            postprocessed_dataset.write_parquet(str(self.config.output_path))

            # Step 8: Print samples
            print("\n" + "=" * 70)
            print("SAMPLE CAPTIONS")
            print("=" * 70)

            for i, sample in enumerate(postprocessed_dataset.take(3)):
                print(f"\nSample {i+1}:")
                print(f"  Image Path: {sample["image"]["path"]}")
                print(f"  Caption: {sample["caption"]}")
                print("-" * 70)

            # Step 9: Calculate metrics
            total_time = time.time() - start_time
            total_samples = postprocessed_dataset.count()

            self.logger.info("Pipeline completed successfully!")

            return {
                "status": "success",
                "total_samples": total_samples,
                "total_time": total_time,
                "inference_time": inference_time,
                "throughput": total_samples / inference_time,
                "output_path": str(self.config.output_path),
            }

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            # Cleanup
            ray.shutdown()
            gc.collect()


def main():
    """Main entry point."""
    # Create configuration
    config = JobConfig(
        dataset_split="train[:100000]",      # 100K images
        num_inference_engines=2,             # Use 2 GPUs
        batch_size=32,                       # vLLM batch size
        preprocessing_batch_size=32,         # Preprocessing batch size
        postprocessing_batch_size=64,        # Postprocessing batch size
        num_partitions=256,                  # More partitions for 100K dataset
        preprocessing_concurrency=16,        # Increased concurrency
        postprocessing_concurrency=16,       # Increased concurrency
        output_filename="02_captioned_dataset_100k.parquet",
    )

    # Create and run pipeline
    pipeline = ImageCaptionPipelineV2(config)
    results = pipeline.run()

    # Print results
    print("\n" + "=" * 70)
    print("PIPELINE RESULTS")
    print("=" * 70)
    print(f"Total samples: {results['total_samples']}")
    print(f"Total time: {results['total_time']:.2f} seconds")
    print(f"Inference time: {results['inference_time']:.2f} seconds")
    print(f"Throughput: {results['throughput']:.2f} samples/second")
    print(f"Output saved to: {results['output_path']}")


if __name__ == "__main__":
    # Ensure HF token is set
    if "HF_TOKEN" not in os.environ:
        raise ValueError("HF_TOKEN is not set")
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "/mnt/data/hf_cache"

    main()
