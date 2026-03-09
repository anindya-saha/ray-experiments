"""
Image Captioning Demo using Ray Data, Ray Data LLM, and vLLM.

- Load dataset
- Dataset repartitioning
- Scale out Preprocess dataset using Qwen 2.5 VL processor.
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

from transformers import Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info


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
    preprocessing_num_cpus: int = 1
    postprocessing_concurrency: int = 8
    postprocessing_num_cpus: int = 1

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
    output_filename: str = "03_captioned_dataset.parquet"

    def __post_init__(self):
        """Ensure output directory exists."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_path(self) -> Path:
        """Full path to output file."""
        return self.output_dir / self.output_filename


class ImageCaptionPipelineV3:
    """Production-ready image captioning pipeline using Ray Data and vLLM."""

    def __init__(self, config: JobConfig):
        """Initialize the pipeline with configuration."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def caption_preprocess(self, row: dict[str, Any]) -> dict[str, Any]:
        """
        Prepare image and prompt for image captioning

        Args:
            row: Input row containing image data
                Example structure:
                {
                    "image": {
                        "bytes": bytes,
                        "path": str,
                    },
                    "label": int,
                }

        Returns:
            Preprocessed row with messages and preserved image data
        """
        # Handle different image schemas (cached vs HuggingFace)
        if isinstance(row["image"], dict):
            # HuggingFace format with bytes
            pil_image = Image.open(BytesIO(row["image"]["bytes"]))
            image_path = row["image"].get("path", "unknown")
            image_bytes = row["image"]["bytes"]
        else:
            # Direct PIL Image from cached dataset
            pil_image = row["image"]
            image_path = "unknown"
            # Convert PIL to bytes for preservation
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        chat_messages = [
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
        # Load processor from pretrained model
        processor = Qwen2_5_VLProcessor.from_pretrained(
            self.config.model_name,
            # min_pixels=self.config.min_pixels,
            # max_pixels=self.config.max_pixels,
        )

        # Preparation for inference
        prompt = processor.tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(chat_messages)

        # print(f"Images: {image_inputs}")
        # print(f"Videos: {video_inputs}")
        # print(f"Prompt: {prompt}")

        return dict(
            # messages=chat_messages, # not needed, since we apply_chat_template=False in the processor config
            prompt=prompt,
            # multi_modal_data= {"image": images},  # not needed, as # vllm_engine_stage.py:345 will take care of it
            image=image_inputs,  # ray data system internally renames it to `images` in the request object, then re-wraps it back as `{"image": ...}` for vLLM.
            sampling_params=dict(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                # detokenize=True, # We should still detokenize the output. Its True by default.
            ),
            # Preserve data for postprocessing
            image_bytes=image_bytes,
            image_path=image_path,
        )

    def caption_postprocess(self, row: dict[str, Any]) -> dict[str, Any]:
        """
        Extract caption from model output and preserve image data.

        Args:
            row: Row containing generated text and preserved data

        Returns:
            Formatted row with caption and image data
        """
        # Reconstruct image dict for visualization
        image_dict = dict(
            bytes=row["image_bytes"],
            path=row["image_path"],
        )

        return dict(
            caption=row["generated_text"],
            image=image_dict,
        )

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
        """Apply preprocessing with parallelization."""
        self.logger.info("Starting parallel preprocessing...")
        preprocessed = dataset.map(
            self.caption_preprocess,
            num_cpus=self.config.preprocessing_num_cpus,
            concurrency=self.config.preprocessing_concurrency,
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
                "mm_processor_kwargs": {
                    "max_pixels": self.config.max_pixels,
                    "min_pixels": self.config.min_pixels,
                },
                "trust_remote_code": True,
                "gpu_memory_utilization": 0.85,
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
            apply_chat_template=False,  # Already applied in preprocessing step
            tokenize=False,  # Already tokenized in preprocessing step
            has_image=False,  # Already handled in preprocessing step
        )

        return build_llm_processor(
            processor_config,
            preprocess=None,
            postprocess=None,
        )

    def postprocess_dataset(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
        """Apply postprocessing with parallelization."""
        self.logger.info("Starting parallel postprocessing...")
        postprocessed = dataset.map(
            self.caption_postprocess,
            num_cpus=self.config.postprocessing_num_cpus,
            concurrency=self.config.postprocessing_concurrency,
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
            }
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
        dataset_split="train[:10000]",      # Small sample for demo
        num_inference_engines=2,            # Number of llm engines to run in parallel
        num_partitions=64,
        preprocessing_concurrency=16,
    )

    # Create and run pipeline
    pipeline = ImageCaptionPipelineV3(config)
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
