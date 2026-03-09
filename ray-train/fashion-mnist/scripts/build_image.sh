#!/bin/bash
set -euo pipefail

CURRENT_USER=${USER:-${USERNAME:-${LOGNAME}}}
if [ -z "${CURRENT_USER+x}" ]; then
  echo "Error: unable to determine username. Set USER, USERNAME, or LOGNAME."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAY_TRAIN_ROOT="$(cd "${EXPERIMENT_DIR}/.." && pwd)"

# -- Defaults -----------------------------------------------------------------

REGISTRY=<registry>
BASE_REPO="${REGISTRY}/${CURRENT_USER}/ray-train-base"
EXPERIMENT_REPO="${REGISTRY}/${CURRENT_USER}/ray-train-fashion-mnist"
PUSH=true
BUILD_BASE=true

# -- Usage --------------------------------------------------------------------

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Build and optionally push Docker images for the Ray Train Fashion-MNIST experiment.

Options:
  --registry URL            Container registry (default: <registry>)
  --base-repo REPO          Base image repository (default: <registry>/<user>/ray-train-base)
  --experiment-repo REPO    Experiment image repository (default: <registry>/<user>/ray-train-fashion-mnist)
  --no-push                 Build only, do not push to registry
  --skip-base               Skip base image build (reuse latest local base)
  -h, --help                Show this help

Outputs (written to stdout on the last two lines):
  BASE_URI=<base-image-uri>
  IMAGE_URI=<experiment-image-uri>

Examples:
  # Build and push both images
  $0

  # Build only, no push
  $0 --no-push

  # Rebuild experiment image only (reuse existing base)
  $0 --skip-base

  # Use in run_ray_job.sh via eval
  eval \$($0)
EOF
  exit 1
}

# -- Parse args ---------------------------------------------------------------

while [[ $# -gt 0 ]]; do
  case $1 in
    --registry)          REGISTRY="$2"; shift 2 ;;
    --base-repo)         BASE_REPO="$2"; shift 2 ;;
    --experiment-repo)   EXPERIMENT_REPO="$2"; shift 2 ;;
    --no-push)           PUSH=false; shift ;;
    --skip-base)         BUILD_BASE=false; shift ;;
    -h|--help)           usage ;;
    *)                   echo "Unknown option: $1"; usage ;;
  esac
done

# -- Build base image ---------------------------------------------------------

if [[ "${BUILD_BASE}" == "true" ]]; then
  echo "==> Building base image..." >&2
  docker build -f "${RAY_TRAIN_ROOT}/docker/Dockerfile.base" \
    -t "${BASE_REPO}:temp" \
    "${RAY_TRAIN_ROOT}" >&2

  BASE_SHA=$(docker inspect --format='{{.Id}}' "${BASE_REPO}:temp")
  BASE_SHA="${BASE_SHA#sha256:}"
  BASE_URI="${BASE_REPO}:${BASE_SHA}"
  docker tag "${BASE_REPO}:temp" "${BASE_URI}" >&2
  docker rmi "${BASE_REPO}:temp" >/dev/null 2>&1 || true
else
  BASE_URI=$(docker images "${BASE_REPO}" --format '{{.CreatedAt}}\t{{.Repository}}:{{.Tag}}' | sort -r | head -1 | cut -f2)
  if [[ -z "${BASE_URI}" ]]; then
    echo "Error: no local base image found for ${BASE_REPO}. Run without --skip-base first." >&2
    exit 1
  fi
  echo "==> Reusing base image: ${BASE_URI}" >&2
fi

# -- Build experiment image ----------------------------------------------------

echo "==> Building experiment image..." >&2
docker build -f "${EXPERIMENT_DIR}/Dockerfile" \
  --build-arg BASE_IMAGE="${BASE_URI}" \
  -t "${EXPERIMENT_REPO}:temp" \
  "${RAY_TRAIN_ROOT}" >&2

IMAGE_SHA=$(docker inspect --format='{{.Id}}' "${EXPERIMENT_REPO}:temp")
IMAGE_SHA="${IMAGE_SHA#sha256:}"
IMAGE_URI="${EXPERIMENT_REPO}:${IMAGE_SHA}"
docker tag "${EXPERIMENT_REPO}:temp" "${IMAGE_URI}" >&2
docker rmi "${EXPERIMENT_REPO}:temp" >/dev/null 2>&1 || true

# -- Push ----------------------------------------------------------------------

if [[ "${PUSH}" == "true" ]]; then
  echo "==> Pushing images..." >&2
  if [[ "${BUILD_BASE}" == "true" ]]; then
    docker push "${BASE_URI}" >&2
  fi
  docker push "${IMAGE_URI}" >&2
fi

# -- Output image URIs (machine-readable) --------------------------------------

echo "==> Base image:      ${BASE_URI}" >&2
echo "==> Experiment image: ${IMAGE_URI}" >&2

echo "BASE_URI=${BASE_URI}"
echo "IMAGE_URI=${IMAGE_URI}"
