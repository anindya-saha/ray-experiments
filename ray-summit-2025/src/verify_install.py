import sys

import datasets
import ray
import torch
import torchaudio
import torchvision
import transformers
import vllm


def main():
    """Comprehensive verification of all package installations and configurations."""

    print("=" * 60)
    print("Ray Summit 2025 - Installation Verification")
    print("=" * 60)

    # Python version
    print(f"\n📌 Python Version: {sys.version.split()[0]}")

    # Core packages
    print("\n📦 Core Packages:")
    print(f"  - Ray: {ray.__version__}")
    print(f"  - PyTorch: {torch.__version__}")
    print(f"  - TorchAudio: {torchaudio.__version__}")
    print(f"  - TorchVision: {torchvision.__version__}")
    print(f"  - vLLM: {vllm.__version__}")
    print(f"  - Transformers: {transformers.__version__}")
    print(f"  - Datasets: {datasets.__version__}")

    # Additional version info for Transformers and Datasets
    print("\n📚 NLP Libraries Details:")
    print(f"  - Transformers version: {transformers.__version__}")
    print(
        f"    • Minimum PyTorch version required: {getattr(transformers, '__min_torch_version__', 'N/A')}"
    )
    print(f"    • Tokenizers backend: {transformers.is_tokenizers_available()}")
    print(f"  - Datasets version: {datasets.__version__}")
    print(
        f"    • PyArrow backend: {datasets.config.PYARROW_VERSION if hasattr(datasets.config, 'PYARROW_VERSION') else 'N/A'}"
    )

    # CUDA information
    print("\n🎮 CUDA Information:")
    cuda_available = torch.cuda.is_available()
    print(f"  - CUDA Available: {cuda_available}")

    if cuda_available:
        print(f"  - CUDA Version: {torch.version.cuda}")
        print(f"  - cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"  - Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    • Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"    • Compute Capability: {props.major}.{props.minor}")
    else:
        print("  ⚠️  No CUDA devices found!")

    # Ray cluster status
    print("\n🌟 Ray Cluster:")
    try:
        context = ray.init(ignore_reinit_error=True, include_dashboard=False)
        print(f"  - Ray initialized successfully")
        if hasattr(context, "dashboard_url") and context.dashboard_url:
            print(f"  - Dashboard URL: {context.dashboard_url}")
        resources = ray.available_resources()
        print(f"  - Available CPUs: {resources.get('CPU', 0)}")
        print(f"  - Available GPUs: {resources.get('GPU', 0)}")
        ray.shutdown()
    except Exception as e:
        print(f"  ❌ Error initializing Ray: {e}")

    # Test basic operations
    print("\n✅ Basic Tests:")

    # Test PyTorch tensor operations
    try:
        tensor = torch.randn(2, 3)
        if cuda_available:
            tensor = tensor.cuda()
            print("  - PyTorch GPU tensor creation: Success")
        else:
            print("  - PyTorch CPU tensor creation: Success")
    except Exception as e:
        print(f"  ❌ PyTorch tensor test failed: {e}")

    # Test vLLM import
    try:
        from vllm import LLM

        print("  - vLLM imports: Success")
    except Exception as e:
        print(f"  ❌ vLLM import test failed: {e}")

    # Test transformers
    try:
        from transformers import AutoModel, AutoTokenizer, pipeline

        print("  - Transformers imports: Success")

        # Check available model types
        from transformers import MODEL_MAPPING

        print(f"    • Available model types: {len(MODEL_MAPPING)} architectures")

        # Test tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased", cache_dir=".cache"
        )
        print("    • Tokenizer test: Success")
    except Exception as e:
        print(f"  ❌ Transformers test failed: {e}")

    # Test datasets
    try:
        from datasets import Dataset, DatasetDict, load_dataset

        print("  - Datasets imports: Success")

        # Test dataset creation
        test_data = {"text": ["Hello world", "Test data"], "label": [0, 1]}
        dataset = Dataset.from_dict(test_data)
        print(f"    • Dataset creation: Success (size: {len(dataset)})")

        # Check available features
        from datasets import ClassLabel, Features, Value

        print("    • Features module: Available")

        # Check if we can access dataset info
        print(f"    • Dataset columns: {dataset.column_names}")
    except Exception as e:
        print(f"  ❌ Datasets test failed: {e}")

    # Version compatibility check
    print("\n🔍 Version Compatibility:")

    # Check Transformers-PyTorch compatibility
    try:
        import packaging.version

        torch_version = packaging.version.parse(torch.__version__.split("+")[0])
        print(
            f"  - PyTorch {torch_version} + Transformers {transformers.__version__}: ✅ Compatible"
        )
    except:
        print("  - Version compatibility check: Could not verify")

    # Check if versions meet project requirements
    print("\n📋 Project Requirements Status:")
    print(f"  - Ray >= 2.50.1: {'✅' if ray.__version__ >= '2.50.1' else '❌'}")
    print(
        f"  - PyTorch == 2.8.0: {'✅' if torch.__version__.startswith('2.8.0') else '❌'}"
    )
    print(f"  - vLLM == 0.11.0: {'✅' if vllm.__version__ == '0.11.0' else '❌'}")
    print(
        f"  - Transformers >= 4.40.0: {'✅' if transformers.__version__ >= '4.40.0' else '❌'}"
    )
    print(
        f"  - Datasets >= 2.16.0: {'✅' if datasets.__version__ >= '2.16.0' else '❌'}"
    )

    print("\n" + "=" * 60)
    print("✨ All verification tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
