#!/bin/bash
set -euo pipefail

CURRENT_USER=${USER:-${USERNAME:-${LOGNAME}}}
if [ -z "${CURRENT_USER+x}" ]; then
  echo "Error: unable to determine username. Set USER, USERNAME, or LOGNAME."
  exit 1
fi

if ! command -v jinja2 &>/dev/null; then
  echo "Error: jinja2 CLI not found. Install with: pip install jinja2-cli"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAY_TRAIN_ROOT="$(cd "${EXPERIMENT_DIR}/.." && pwd)"
TEMPLATE_FILE="${EXPERIMENT_DIR}/k8s/ray_job.template.j2.yaml"
DRY_RUN=false
SKIP_BUILD=false
SKIP_BASE=false

# -- Defaults -----------------------------------------------------------------

REGISTRY=<registry>
EXPERIMENT_REPO="${REGISTRY}/${CURRENT_USER}/ray-train-fashion-mnist"

NAMESPACE=<namespace>
SERVICE_ACCOUNT=<service-account>
FSX_CLAIM="fsx-static-claim"
FSX_MOUNT="/mnt/fsx-static"

HEAD_CPU="4"
HEAD_MEMORY="8Gi"
GPU_PER_WORKER="1"

declare -A NODE_PRESETS=(
  [p4d]="ml.p4d.24xlarge"
  [p4de]="ml.p4de.24xlarge"
  [p5en]="ml.p5en.48xlarge"
  [p6]="ml.p6-b200.48xlarge"
)

# -- Usage --------------------------------------------------------------------

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Launch a Ray Train job on Kubernetes.

Required:
  --job-name NAME           Job name (used as RayJob metadata.name)
  --node-type TYPE          Node type preset: p4d, p4de, p5en, p6

Optional:
  --num-workers NUM         Number of GPU workers (default: 2)
  --gpu-per-worker NUM      GPUs per worker (default: 1)
  --config-file FILE        Training config file path inside container
                            (default: /home/ray/fashion-mnist/configs/config.yaml)
  --override KEY=VALUE      OmegaConf override (repeatable)
  --namespace NS            Kubernetes namespace 
  --service-account SA      Service account

  --data-dir PATH           Data directory on FSx (default: derived from user)
  --storage-path PATH       Checkpoint storage path on FSx (default: derived from user)

  --head-cpu NUM            Head pod CPU (default: 4)
  --head-memory SIZE        Head pod memory (default: 8Gi)

  --skip-build              Skip all Docker builds (reuse existing image)
  --skip-base               Skip base image build, rebuild experiment image only
  --dry-run                 Render YAML to stdout without submitting

  -h, --help                Show this help

Examples:
  # Basic 2-worker FSDP run
  $0 --job-name fsdp-test --node-type p5en --num-workers 2

  # 4-worker run with overrides
  $0 --job-name fsdp-4w --node-type p5en --num-workers 4 \\
     --override training.epochs=5 \\
     --override training.strategy=fsdp

  # Preview YAML without submitting
  $0 --job-name preview --node-type p5en --dry-run
EOF
  exit 1
}

# -- Parse args ---------------------------------------------------------------

NUM_WORKERS="2"
CONFIG_FILE="/home/ray/fashion-mnist/configs/config.yaml"
OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --job-name)        JOB_NAME="$2"; shift 2 ;;
    --node-type)       NODE_TYPE_INPUT="$2"; shift 2 ;;
    --num-workers)     NUM_WORKERS="$2"; shift 2 ;;
    --gpu-per-worker)  GPU_PER_WORKER="$2"; shift 2 ;;
    --config-file)     CONFIG_FILE="$2"; shift 2 ;;
    --override)        OVERRIDES+=("$2"); shift 2 ;;
    --namespace)       NAMESPACE="$2"; shift 2 ;;
    --service-account) SERVICE_ACCOUNT="$2"; shift 2 ;;
    --data-dir)        DATA_DIR="$2"; shift 2 ;;
    --storage-path)    STORAGE_PATH="$2"; shift 2 ;;
    --head-cpu)        HEAD_CPU="$2"; shift 2 ;;
    --head-memory)     HEAD_MEMORY="$2"; shift 2 ;;
    --skip-build)      SKIP_BUILD=true; shift ;;
    --skip-base)       SKIP_BASE=true; shift ;;
    --dry-run)         DRY_RUN=true; shift ;;
    -h|--help)         usage ;;
    *)                 echo "Unknown option: $1"; usage ;;
  esac
done

# -- Validate -----------------------------------------------------------------

if [[ -z ${JOB_NAME:-} ]]; then
  echo "Error: --job-name is required"
  usage
fi

if [[ -z ${NODE_TYPE_INPUT:-} ]]; then
  echo "Error: --node-type is required"
  usage
fi

INSTANCE_TYPE="${NODE_PRESETS[$NODE_TYPE_INPUT]:-$NODE_TYPE_INPUT}"

# Derive FSx paths from username if not provided
DATA_DIR="${DATA_DIR:-${FSX_MOUNT}/${CURRENT_USER}/ray-experiments/ray-train/fashion_mnist/data}"
STORAGE_PATH="${STORAGE_PATH:-${FSX_MOUNT}/${CURRENT_USER}/ray-experiments/ray-train/fashion_mnist/checkpoints}"

# Inject infra overrides
OVERRIDES+=("training.data_dir=${DATA_DIR}")
OVERRIDES+=("ray.run.storage_path=${STORAGE_PATH}")
OVERRIDES+=("ray.scaling.num_workers=${NUM_WORKERS}")

# -- Docker build & push ------------------------------------------------------

if [[ "${SKIP_BUILD}" == "false" ]]; then
  BUILD_ARGS=()
  if [[ "${SKIP_BASE}" == "true" ]]; then
    BUILD_ARGS+=("--skip-base")
  fi
  eval "$("${SCRIPT_DIR}/build_image.sh" "${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}")"
else
  IMAGE_URI=$(docker images "${EXPERIMENT_REPO}" --format '{{.CreatedAt}}\t{{.Repository}}:{{.Tag}}' | sort -r | head -1 | cut -f2)
  if [[ -z "${IMAGE_URI}" ]]; then
    echo "Error: no local image found for ${EXPERIMENT_REPO}. Run without --skip-build first."
    exit 1
  fi
  echo "==> Reusing image: ${IMAGE_URI}"
fi

# -- Build JSON context -------------------------------------------------------

OVERRIDES_JSON=$(printf '%s\n' "${OVERRIDES[@]}" | jq -R . | jq -s .)

read -r -d '' CONTEXT_JSON <<EOF || true
{
  "JOB_NAME": "${JOB_NAME}",
  "NAMESPACE": "${NAMESPACE}",
  "SERVICE_ACCOUNT": "${SERVICE_ACCOUNT}",
  "INSTANCE_TYPE": "${INSTANCE_TYPE}",
  "IMAGE_URI": "${IMAGE_URI}",
  "NUM_WORKERS": ${NUM_WORKERS},
  "GPU_PER_WORKER": ${GPU_PER_WORKER},
  "HEAD_CPU": "${HEAD_CPU}",
  "HEAD_MEMORY": "${HEAD_MEMORY}",
  "DATA_DIR": "${DATA_DIR}",
  "STORAGE_PATH": "${STORAGE_PATH}",
  "FSX_CLAIM": "${FSX_CLAIM}",
  "OVERRIDES": ${OVERRIDES_JSON}
}
EOF

# -- Render template -----------------------------------------------------------

YAML_OUTPUT_FILE="${PWD}/${JOB_NAME}.yaml"

jinja2 "${TEMPLATE_FILE}" <(echo "${CONTEXT_JSON}") --format=json > "${YAML_OUTPUT_FILE}"
echo "==> Generated YAML: ${YAML_OUTPUT_FILE}"

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "---"
  cat "${YAML_OUTPUT_FILE}"
  exit 0
fi

# -- Submit to K8s -------------------------------------------------------------

if kubectl get rayjob "${JOB_NAME}" -n "${NAMESPACE}" &>/dev/null; then
  JOB_STATUS=$(kubectl get rayjob "${JOB_NAME}" -n "${NAMESPACE}" \
    -o jsonpath='{.status.jobStatus}' 2>/dev/null || echo "")

  if [[ "${JOB_STATUS}" == "SUCCEEDED" || "${JOB_STATUS}" == "FAILED" || "${JOB_STATUS}" == "STOPPED" ]]; then
    echo "==> Cleaning up finished RayJob '${JOB_NAME}'..."
    kubectl delete rayjob "${JOB_NAME}" -n "${NAMESPACE}" --wait=true
  else
    echo "Error: RayJob '${JOB_NAME}' already exists and is still active (status: ${JOB_STATUS:-Unknown})."
    echo "Delete it first: kubectl delete rayjob ${JOB_NAME} -n ${NAMESPACE}"
    exit 1
  fi
fi

echo "==> Submitting RayJob..."
kubectl create -f "${YAML_OUTPUT_FILE}"
echo "==> RayJob '${JOB_NAME}' submitted to namespace '${NAMESPACE}'"

echo ""
echo "==> Useful commands"
echo "    Status:    kubectl get rayjob ${JOB_NAME} -n ${NAMESPACE}"
echo "    Delete:    kubectl delete rayjob ${JOB_NAME} -n ${NAMESPACE}"
echo ""
echo "==> Ray Dashboard (once cluster is running)"
echo "    HEAD_SVC=\$(kubectl get svc -n ${NAMESPACE} -l ray.io/node-type=head -o name | grep ${JOB_NAME})"
echo "    kubectl port-forward \$HEAD_SVC 8265:8265 -n ${NAMESPACE}"
echo "    Then open: http://localhost:8265"
