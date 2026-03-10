#!/bin/bash
#SBATCH --job-name=vllm-pd
#SBATCH -N 4
#SBATCH -n 4
#SBATCH --ntasks-per-node=1
#SBATCH --spread-job
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --output="/tmp/vllm-pd-slurm-%j.out"
#SBATCH --error="/tmp/vllm-pd-slurm-%j.err"

echo "=== Job Start Time ==="
echo "UTC Time: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo "======================="
echo ""

# Define valid model names
VALID_MODELS=( \
    "Llama-3.1-405B-Instruct-FP8-KV" \
    "amd-Llama-3.3-70B-Instruct-FP8-KV" \
    "DeepSeek-V3" \
    "DeepSeek-R1-0528" \
    "gpt-oss-120b" \
)

declare -A MODEL_RUNFILES=(
    ["Llama-3.1-405B-Instruct-FP8-KV"]="vllm_disagg_server.sh"
    ["amd-Llama-3.3-70B-Instruct-FP8-KV"]="vllm_disagg_server.sh"
    ["DeepSeek-V3"]="vllm_disagg_server.sh"
    ["DeepSeek-R1-0528"]="vllm_disagg_server.sh"
    ["gpt-oss-120b"]="vllm_disagg_server.sh"
)

# HF cache directory names differ from MODEL_NAME
declare -A MODEL_DIR_NAMES=(
    ["DeepSeek-R1-0528"]="models--deepseek-ai--DeepSeek-R1-0528"
)

if [[ -n "${MODEL_RUNFILES[$MODEL_NAME]}" ]]; then
    RUN_FILE="${MODEL_RUNFILES[$MODEL_NAME]}"
    echo "Model found: $MODEL_NAME -> $RUN_FILE"
else
    echo "Error: Model '$MODEL_NAME' not found. Available: ${!MODEL_RUNFILES[@]}"
    exit 1
fi

export DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-vllm_disagg_pd:latest}"
if test -z "${DOCKER_IMAGE_NAME}"; then
    echo "Error: DOCKER_IMAGE_NAME is not set or empty."; exit 1
fi

# Use NFS path so all nodes can access the scripts
NIXL_REPO_DIR="${NIXL_REPO_DIR:-/nfsdata/vllm_disagg_scripts}"
LOG_PATH="${LOG_PATH:-/tmp/vllm_disagg_logs}"

xP="${xP:-1}"
yD="${yD:-1}"
MODEL_NAME="${MODEL_NAME:-None}"
MODEL_DIR="${MODEL_DIR:-/nfsdata/hf_hub_cache-0}"

validate_model_name() {
    for model in "${VALID_MODELS[@]}"; do
        [[ "$MODEL_NAME" == "$model" ]] && return 0
    done
    printf "Error: Invalid MODEL_NAME: '%s'\nValid models:\n" "$MODEL_NAME"
    for model in "${VALID_MODELS[@]}"; do printf "  - %s\n" "$model"; done
    exit 1
}
validate_model_name

# ------------------------
# Model path resolution
# ------------------------
DISK_DIR_NAME="${MODEL_DIR_NAMES[$MODEL_NAME]:-$MODEL_NAME}"
echo "Looking for model: $MODEL_NAME (disk dir: $DISK_DIR_NAME)"

# Resolve HF cache snapshot path
resolve_hf_cache_path() {
    local base_path=$1
    if [[ -d "${base_path}/snapshots" ]]; then
        local snapshot=$(ls -1 "${base_path}/snapshots" 2>/dev/null | head -1)
        if [[ -n "$snapshot" ]]; then
            echo "${base_path}/snapshots/${snapshot}"
            return 0
        fi
    fi
    echo "$base_path"
    return 1
}

MODEL_PATH=""
SEARCH_PATHS=(
    "/nfsdata/hf_hub_cache-0/${DISK_DIR_NAME}"
    "/mnt/m2m_nobackup/models_blog/${DISK_DIR_NAME}"
    "/shared_inference/models_blog/${DISK_DIR_NAME}"
    "${MODEL_DIR}/${DISK_DIR_NAME}"
    "/nfsdata/hf_hub_cache-0/${MODEL_NAME}"
    "/shared_inference/models_blog/${MODEL_NAME}"
    "${MODEL_DIR}/${MODEL_NAME}"
)

for search_path in "${SEARCH_PATHS[@]}"; do
    if [[ -d "$search_path" ]]; then
        RESOLVED=$(resolve_hf_cache_path "$search_path")
        MODEL_PATH="$RESOLVED"
        echo "✓ Found MODEL_PATH: $MODEL_PATH"
        break
    fi
done

if [[ -z "$MODEL_PATH" ]]; then
    echo "FATAL: Model '$MODEL_NAME' not found. Searched:"
    for p in "${SEARCH_PATHS[@]}"; do echo "  - $p"; done
    exit 1
fi
echo "Final MODEL_PATH: $MODEL_PATH"

# Calculate NUM_NODES
NUM_NODES=$((xP + yD + 1))
echo "NUM_NODES: $NUM_NODES (xP=$xP + yD=$yD + 1)"

# ------------------------
# Node selection from SLURM allocation
# ------------------------
FULL_NODELIST=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
SELECTED_NODES=$(echo "$FULL_NODELIST" | head -n $NUM_NODES)

export SLURM_NNODES=$NUM_NODES
export SLURM_NTASKS=$NUM_NODES
export SLURM_JOB_NUM_NODES=$NUM_NODES
export SLURM_NPROCS=$NUM_NODES
NEW_SLURM_NODELIST=$(echo "$SELECTED_NODES" | paste -sd,)
export SLURM_JOB_NODELIST="$NEW_SLURM_NODELIST"
export SLURM_NODELIST="$NEW_SLURM_NODELIST"
export SLURM_TASKS_PER_NODE="1(x$NUM_NODES)"
export SLURM_NTASKS_PER_NODE=1

echo "Selected nodes: $NEW_SLURM_NODELIST"

# Resolve IPs
USER_NAME=$(whoami)
MASTER_NODE=$(echo "$SELECTED_NODES" | head -n 1)
# Use default route source IP (management network) for inter-node TCP communication
MASTER_ADDR=$(srun --chdir=/tmp --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="$MASTER_NODE" bash -c 'ip route get 1.1.1.1 | sed -n "s/.*src \([^ ]*\).*/\1/p"')
MASTER_ADDR=$(echo "$MASTER_ADDR" | head -1 | tr -d '[:space:]')
MASTER_PORT=39566

IPS=()
for NODE in $SELECTED_NODES; do
    IP=$(srun --chdir=/tmp --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="$NODE" bash -c 'ip route get 1.1.1.1 | sed -n "s/.*src \([^ ]*\).*/\1/p"')
    IP=$(echo "$IP" | head -1 | tr -d '[:space:]')
    IPS+=("$IP")
done

echo "Node IPs: ${IPS[*]}"

NIXL_COOKBOOK_PATH="/opt/nixl-vllm-cookbook"
NNODES=$NUM_NODES

echo "MASTER_NODE=$MASTER_NODE  MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"

mkdir -p "$LOG_PATH"

export LOG_PATH NIXL_REPO_DIR NIXL_COOKBOOK_PATH NNODES MASTER_ADDR MASTER_PORT MODEL_PATH
export xP yD MODEL_NAME USER_NAME
export IPADDRS="$(echo "${IPS[*]}" | sed 's/ /,/g')"
export DOCKER_CONT_NAME="container_${MODEL_NAME}_${SLURM_JOB_ID}"
export RUN_FILE_FULL="$NIXL_COOKBOOK_PATH/${RUN_FILE}"

SELECTED_NODELIST_SRUN=$(echo "$SELECTED_NODES" | paste -sd,)

srun --chdir=/tmp --nodelist="$SELECTED_NODELIST_SRUN" bash -c '
echo "Rank $SLURM_PROCID on $(hostname)";
sudo docker ps -q | xargs --no-run-if-empty sudo docker stop;
sudo docker run --rm \
    --device /dev/dri \
    --device /dev/kfd \
    --device /dev/infiniband \
    --network host \
    --ipc host \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    -v $HOME:$HOME \
    -v /nfsdata:/nfsdata \
    -v /shared_inference:/shared_inference \
    -v /mnt/m2m_nobackup:/mnt/m2m_nobackup \
    -v $HOME/.ssh:/root/.ssh \
    --shm-size 64G \
    --ulimit nofile=131072:131072 \
    -v ${LOG_PATH}:/run_logs \
    -v $NIXL_REPO_DIR:$NIXL_COOKBOOK_PATH \
    -e SLURM_JOB_ID=$SLURM_JOB_ID \
    -e SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST \
    -e NNODES=$NNODES \
    -e NODE_RANK=$SLURM_PROCID \
    -e MASTER_ADDR=$MASTER_ADDR \
    -e MASTER_PORT=$MASTER_PORT \
    -e MODEL_PATH=$MODEL_PATH \
    -e NIXL_COOKBOOK_PATH=$NIXL_COOKBOOK_PATH \
    -e xP=$xP \
    -e yD=$yD \
    -e USER_NAME=$USER_NAME \
    -e MODEL_NAME=$MODEL_NAME \
    -e IPADDRS=$IPADDRS \
    -e PROXY_TYPE=${PROXY_TYPE:-vllm_router} \
    -e ROUTER_PORT=${ROUTER_PORT:-2584} \
    -e BENCHMARK_PORT=${BENCHMARK_PORT:-2584} \
    -e DRY_RUN=${DRY_RUN:-0} \
    -e UCX_TLS=all \
    -e UCX_SOCKADDR_TLS_PRIORITY=tcp \
    -e UCX_MEMTYPE_CACHE=y \
    -e UCX_RNDV_SCHEME=get_zcopy \
    -e UCX_RNDV_THRESH=4k \
    -e UCX_ROCM_IPC_MIN_ZCOPY=0 \
    -e UCX_LOG_LEVEL=info \
    -e HSA_ENABLE_SDMA=1 \
    --name $DOCKER_CONT_NAME \
    $DOCKER_IMAGE_NAME bash -c "
        mkdir -p /run_logs/${SLURM_JOB_ID}
        $RUN_FILE_FULL 2>&1 | tee /run_logs/${SLURM_JOB_ID}/pd_vllm_bench_NODE${SLURM_PROCID}.log
    "
'
srun --chdir=/tmp --nodelist="$SELECTED_NODELIST_SRUN" bash -c 'sudo docker stop $DOCKER_CONT_NAME 2>/dev/null; sudo docker rm $DOCKER_CONT_NAME 2>/dev/null'
