# Developer Guide

## Project Structure with UV

This project uses [UV](https://github.com/astral-sh/uv) as the Python package and project manager. UV provides fast, reliable Python package management with a focus on reproducibility.

This project includes:
- **Ray 2.50.1** - Distributed computing framework
- **PyTorch 2.8.0** with CUDA 12.8 support
- **vLLM 0.11.0** - High-performance LLM inference
- **Transformers 4.57.1** - Hugginface Transformer State-of-the-art NLP models
- **Datasets >= 4.0.0** - Hugginface Transformer datasets

### Key UV Files

- **`pyproject.toml`** - Project configuration and dependencies
  - Main dependencies (ray, torch, vllm, etc.)
  - Optional dev dependencies (black, isort, flake8)
  - Tool configurations (black, isort settings)
  - UV-specific configurations (PyTorch CUDA index)

- **`uv.lock`** - Locked dependency versions
  - Ensures reproducible installations across environments
  - Contains exact versions and hashes of all dependencies
  - Automatically updated when dependencies change

- **`.python-version`** - Python version specification
  - Specifies Python 3.12 for this project
  - UV automatically uses this version when creating virtual environments

### UV Commands Used in This Project

```bash
# Install all dependencies
uv sync

# Install with dev dependencies
uv sync --extra dev

# Add a new package
uv add package-name

# Remove a package
uv remove package-name

# Update dependencies
uv lock --upgrade
```

### Virtual Environment Management

UV automatically creates and manages a virtual environment in `.venv/`.

To activate the virtual environment:
```bash
source .venv/bin/activate
```

### Verification

Run the verification script to check all installations:
```bash
# Run commands in the virtual environment
source .venv/bin/activate

python3 src/verify_install.py
```

## Running the code


```bash
# Run commands in the virtual environment
export HF_TOKEN=<YOUR HF_TOKEN>
export HF_HOME=<some folder where huggingface would download models, datasets etc.>

export HF_TOKEN=
export HF_HOME=/mnt/data/hf_cache
```

```bash
# Run commands in the virtual environment

python3 src/01_image_caption_demo.py
```

**Note:** I am using a AWS g5.12xlarge machine with 4 x A10 GPUs. If you have a smaller machine then you can make change to the following attributes:
```bash
def create_llm_processor(self):
    """Create and configure the vLLM processor."""
    processor_config = vLLMEngineProcessorConfig(
        model_source=...,
        engine_kwargs={
          ...
        },
        runtime_env={
          ...
        },
        ...
        # comment out this line, let Ray figure out
        # accelerator_type=self.config.accelerator_type,
        ...
    )
```

```bash
def main():
    """Main entry point."""
    # Create configuration
    config = JobConfig(
        dataset_split="train[:100]",      # reduce the datatset size to make a small sample for demo
        num_inference_engines=1,          # change the number of llm engines to 1
    )
```


### PyTorch with CUDA Support

The project is configured to use `PyTorch 2.8.0` with `CUDA 12.8` support:

```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cu128", marker = "sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

This ensures GPU-accelerated PyTorch packages are installed on Linux systems.

### Caveat

I encountered a `uvloop` issue:
```bash
  File "/tmp/ray/session_2025-10-19_10-02-47_133969_37536/runtime_resources/working_dir_files/_ray_pkg_98c81e886c19cb78/.venv/lib/python3.12/site-packages/uvloop/__init__.py", line 206, in get_event_loop
    raise RuntimeError(
RuntimeError: There is no current event loop in thread 'MainThread'.
```

The error we encountered `(RuntimeError: There is no current event loop in thread 'MainThread')` occurs because:
+ Ray Data creates actors in separate processes
+ uvloop has stricter requirements for event loop creation
+ When Ray tries to create an event loop in the actor process, uvloop throws an error

I tried by setting `asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())` and `UVLOOP_DISABLE=1` in the env_vars, but that did not help.

So, I disabled installing `uvloop`. Added the following in `pyproject.toml`.
```bash
[tool.uv]
# Error we encountered `(RuntimeError: There is no current event loop in thread 'MainThread')` occurs because:
#   Ray Data creates actors in separate processes
#   uvloop has stricter requirements for event loop creation
#   When Ray tries to create an event loop in the actor process, uvloop throws an error
# Solution: Do not install uvloop
override-dependencies = [
    "uvloop ; sys_platform == 'never'"
]
```

### Development Dependencies

Install development dependencies which includes:
- **Black** - Code formatter
- **isort** - Import sorter
- **flake8** - Linter


```bash
uv sync --extra dev
```

**Formatting Code**


```bash
black .
black src/01_image_caption_demo.py
```

```bash
isort .
isort src/01_image_caption_demo.py
```

**VS Code Setup**

For auto-formatting on save, create `.vscode/settings.json`:
```json
{
    "editor.formatOnSave": true,
    "python.formatting.provider": "none",
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": "explicit"
        }
    },
    "black-formatter.args": [
        "--config",
        "${workspaceFolder}/pyproject.toml"
    ],
    "isort.args": [
        "--settings-path",
        "${workspaceFolder}/pyproject.toml"
    ]
}
```
