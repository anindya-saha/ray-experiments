# Fashion-MNIST Ray Train Experiment

Distributed training on Fashion-MNIST using Ray Train with FSDP or DeepSpeed.

## Project Structure

```
fashion-mnist/
  configs/          Training and DeepSpeed configs
  k8s/              Kubernetes manifests and Jinja2 templates
  scripts/          Build and launch automation
  src/              Training and data download code
  Dockerfile        Experiment image (layers code on top of base)
```

## Docker Images

This experiment uses a two-layer image strategy. The base image holds all shared
dependencies (Ray, PyTorch, DeepSpeed, etc.) and rebuilds only when deps change.
The experiment image layers the training code on top and rebuilds in seconds.

### Build and push (automated)

```bash
# Build both images and push to registry
scripts/build_image.sh

# Build only, no push
scripts/build_image.sh --no-push

# Rebuild experiment image only (reuse existing base)
scripts/build_image.sh --skip-base
```

Run `scripts/build_image.sh --help` for all options.

### Registry login

```bash
docker login <repo>
```

### Test docker image locally

```bash
docker run --rm -it \
    --gpus all \
    --name ray-gpu-container \
    <repo>/${USER}/ray-train-fashion-mnist:<tag> \
    /bin/bash
```

## Local Training

```bash
python src/train.py --config configs/config.yaml
```

Override any config value via CLI:

```bash
python src/train.py --config configs/config.yaml \
    --override training.epochs=5 \
    --override training.strategy=deepspeed
```

## Kubernetes (RayJob)

### Quick launch

The `run_ray_job.sh` script automates the full workflow -- Docker build, push,
Jinja2 template rendering, and `kubectl create`:

```bash
# Basic 2-worker FSDP run on p5en nodes
scripts/run_ray_job.sh --skip-build \
    --job-name fashion-mnist-fsdp \
    --node-type p5en \
    --num-workers 2

# 4-worker run with training overrides
scripts/run_ray_job.sh --skip-base \
    --job-name fashion-mnist-fsdp \
    --node-type p5en \
    --num-workers 4 \
    --override training.epochs=5

# Preview rendered YAML without submitting
scripts/run_ray_job.sh --skip-build \
    --job-name fashion-mnist-fsdp \
    --node-type p5en \
    --dry-run

# Skip Docker rebuild totally (reuse existing image)
scripts/run_ray_job.sh --skip-build \
    --job-name fashion-mnist-fsdp \
    --node-type p5en

# Skip base Docker rebuild; but build experiment image
scripts/run_ray_job.sh --skip-base \
    --job-name fashion-mnist-fsdp \
    --node-type p5en

Run `scripts/run_ray_job.sh --help` for all available options.

**Prerequisites:** `jinja2-cli` (`pip install jinja2-cli`), `jq`, Docker, kubectl.

### Manual deploy

You can still apply the static manifest directly:

```bash
kubectl create -f k8s/ray_job.yaml -n mlp
```

> **Note:** Update `ray_job.yaml` with your actual image tag, FSx/PVC
> mounts, namespace, and resource requests before deploying.
