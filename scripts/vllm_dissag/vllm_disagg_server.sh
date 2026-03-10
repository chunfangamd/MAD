#!/bin/bash
# vLLM Disaggregated Server Launcher with Model-Specific Configurations
# =============================================================================
#
# Node role assignment (by NODE_RANK):
#   0            -> Proxy/Router node
#   1..xP        -> Prefill nodes  (kv_producer)
#   xP+1..xP+yD -> Decode nodes   (kv_consumer)

# =============================================================================
# Environment Configuration
# =============================================================================

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-23731}"
NODE_RANK="${NODE_RANK:-0}"
MODEL_PATH=$MODEL_PATH
MODEL_NAME="${MODEL_NAME:-}"
xP="${xP:-1}"
yD="${yD:-1}"
IPADDRS="${IPADDRS:-localhost}"
DRY_RUN="${DRY_RUN:-0}"

PROXY_TYPE="${PROXY_TYPE:-vllm_router}"
ROUTER_PORT="${ROUTER_PORT:-2584}"

if [[ "$PROXY_TYPE" != "vllm_router" && "$PROXY_TYPE" != "toy_proxy" ]]; then
    echo "Error: Invalid PROXY_TYPE='$PROXY_TYPE'. Must be 'vllm_router' or 'toy_proxy'." >&2
    exit 1
fi

echo "Listing NIXL_COOKBOOK_PATH : "
ls ${NIXL_COOKBOOK_PATH}

# =============================================================================
# Dependencies and Environment Setup
# =============================================================================

pip install py-spy
pip install --ignore-installed --force-reinstall flask

# Patch Nixl UCX backend: set ucx_error_handling_mode=none for shared-memory
# transport compatibility (Pensando ionic NICs don't support rdmacm, so the
# default UCP_ERR_HANDLING_MODE_PEER causes "no active messages transport" errors)
NIXL_API_FILE=$(python3 -c "import rixl._api; print(rixl._api.__file__)" 2>/dev/null)
if [[ -n "$NIXL_API_FILE" ]]; then
    if ! grep -q 'ucx_error_handling_mode' "$NIXL_API_FILE"; then
        sed -i '/init\["num_threads"\] = str(nixl_conf.num_threads)/a\                        init["ucx_error_handling_mode"] = "none"' "$NIXL_API_FILE"
        echo "[PATCH] Added ucx_error_handling_mode=none to $NIXL_API_FILE"
    else
        echo "[PATCH] ucx_error_handling_mode already set in $NIXL_API_FILE"
    fi
fi

source ${NIXL_COOKBOOK_PATH}/env.sh

# Management IP: use default route source (most reliable across clusters)
host_ip=$(ip route get 1.1.1.1 2>/dev/null | awk '/src/ {print $7}')
# RDMA IP: first 192.168.x IP for Nixl KV transfer
rdma_ip=$(hostname -I | tr ' ' '\n' | grep '^192\.168\.' | head -1)
rdma_ip="${rdma_ip:-$host_ip}"
host_name=$(hostname)
SERVER_PORT=2584

echo "[INFO] Management IP (barriers/proxy): $host_ip"
echo "[INFO] RDMA IP (Nixl KV transfer): $rdma_ip"

if [ "$PROXY_TYPE" == "vllm_router" ]; then
    PROXY_PORT=$ROUTER_PORT
else
    PROXY_PORT=$SERVER_PORT
fi

if [[ -z "$UCX_NET_DEVICES" ]]; then
    echo "Error: UCX_NET_DEVICES is empty after env.sh detection" >&2
    exit 1
fi

if [[ -z "$NCCL_SOCKET_IFNAME" ]]; then
    echo "Error: NCCL_SOCKET_IFNAME is empty after env.sh detection" >&2
    exit 1
fi

# =============================================================================
# Model-Specific Configuration Maps
# =============================================================================

declare -A MODEL_PREFILL_CONFIGS=(
    ["Llama-3.1-405B-Instruct-FP8-KV"]="--tensor-parallel-size 8 --kv-cache-dtype fp8"
    ["amd-Llama-3.3-70B-Instruct-FP8-KV"]="--tensor-parallel-size 8 --max-model-len 65536 --kv-cache-dtype fp8"
    ["DeepSeek-V3"]="--tensor-parallel-size 8 --compilation-config '{\"cudagraph_mode\":\"PIECEWISE\"}' --no-enable-prefix-caching --block-size 1"
    ["DeepSeek-R1-0528"]="--tensor-parallel-size 8 --compilation-config '{\"cudagraph_mode\":\"PIECEWISE\"}' --no-enable-prefix-caching --block-size 1"
    ["gpt-oss-120b"]="--tensor-parallel-size 8"
)

declare -A MODEL_DECODE_CONFIGS=(
    ["Llama-3.1-405B-Instruct-FP8-KV"]="--tensor-parallel-size 8 --kv-cache-dtype fp8"
    ["amd-Llama-3.3-70B-Instruct-FP8-KV"]="--tensor-parallel-size 8 --max-model-len 65536 --kv-cache-dtype fp8"
    ["DeepSeek-V3"]="--tensor-parallel-size 8 --compilation-config '{\"cudagraph_mode\":\"PIECEWISE\"}' --no-enable-prefix-caching --block-size 1"
    ["DeepSeek-R1-0528"]="--tensor-parallel-size 8 --compilation-config '{\"cudagraph_mode\":\"PIECEWISE\"}' --no-enable-prefix-caching --block-size 1"
    ["gpt-oss-120b"]="--tensor-parallel-size 8"
)

declare -A MODEL_ENVS=(
    ["amd-Llama-3.3-70B-Instruct-FP8-KV"]="VLLM_USE_V1=1 VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 AMDGCN_USE_BUFFER_OPS=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_RMSNORM=1 VLLM_USE_AITER_TRITON_ROPE=1 TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 TRITON_HIP_USE_ASYNC_COPY=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 TRITON_HIP_ASYNC_FAST_SWIZZLE=1 "
    ["Llama-3.1-405B-Instruct-FP8-KV"]="VLLM_USE_V1=1 VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 AMDGCN_USE_BUFFER_OPS=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_RMSNORM=1 VLLM_USE_AITER_TRITON_ROPE=1 TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 TRITON_HIP_USE_ASYNC_COPY=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 TRITON_HIP_ASYNC_FAST_SWIZZLE=1 "
    ["DeepSeek-V3"]="VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_PAGED_ATTN=0 VLLM_ROCM_USE_AITER_RMSNORM=1 VLLM_USE_AITER_TRITON_SILU_MUL=0 "
    ["DeepSeek-R1-0528"]="VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_PAGED_ATTN=0 VLLM_ROCM_USE_AITER_RMSNORM=1 VLLM_USE_AITER_TRITON_SILU_MUL=0 "
    ["gpt-oss-120b"]="VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_TRITON_BF16_GEMM=0 VLLM_USE_AITER_UNIFIED_ATTENTION=1 VLLM_ROCM_USE_AITER_MHA=0 ROCM_TRITON_MOE_PRESHUFFLE_SCALES=0 "
)

# =============================================================================
# Configuration Selection Functions
# =============================================================================

get_model_config() {
    local mode="$1"
    local model_name="$2"
    if [[ "$mode" == "prefill" ]]; then
        echo "${MODEL_PREFILL_CONFIGS[$model_name]:-"--tp-size 4"}"
    elif [[ "$mode" == "decode" ]]; then
        echo "${MODEL_DECODE_CONFIGS[$model_name]:-"--tp-size 4"}"
    fi
}

get_model_envs() {
    echo "${MODEL_ENVS[$1]:-""}"
}

if [[ -z "$MODEL_NAME" ]]; then
    echo "ERROR: MODEL_NAME is not set"; exit 1
fi

PREFILL_MODEL_CONFIG=$(get_model_config "prefill" "$MODEL_NAME")
DECODE_MODEL_CONFIG=$(get_model_config "decode" "$MODEL_NAME")
PREFILL_MODEL_ENVS=$(get_model_envs "$MODEL_NAME")
DECODE_MODEL_ENVS=$(get_model_envs "$MODEL_NAME")
echo "Using model-specific configuration for: $MODEL_NAME"

ENGINE_ID="${ENGINE_ID:-${MODEL_NAME}-pd-run}"

# =============================================================================
# Container Synchronization
# =============================================================================

echo "Waiting at the container creation barrier on $host_name"
python3 $NIXL_COOKBOOK_PATH/sync.py barrier \
    --local-ip ${host_ip} \
    --local-port 5000 \
    --enable-port \
    --node-ips ${IPADDRS} \
    --node-ports 5000 \
    --wait-for-all-ports \
    --timeout 300

# =============================================================================
# ETCD Server Setup
# =============================================================================

echo "Proceeding to start etcd server on $host_name"
${NIXL_COOKBOOK_PATH}/start_etcd.sh > /dev/null &
etcd_pid=$!

echo "Waiting at etcd server barrier on $host_name"
python3 $NIXL_COOKBOOK_PATH/sync.py barrier \
    --node-ips ${IPADDRS} \
    --node-ports 2379 \
    --wait-for-all-ports \
    --timeout 300

echo "All etcd servers are up : $host_name"
sleep 3

echo "etcd endpoint health=================="
etcdctl endpoint health 2>&1 || /usr/local/bin/etcd/etcdctl endpoint health 2>&1 || true
echo "======================================"
echo "etcd member list======================"
etcdctl member list 2>&1 || /usr/local/bin/etcd/etcdctl member list 2>&1 || true
echo "======================================"

python3 $NIXL_COOKBOOK_PATH/sync.py barrier \
    --node-ips ${IPADDRS} \
    --node-ports 2379 \
    --wait-for-all-ports \
    --timeout 300

# =============================================================================
# Cluster Topology Configuration
# =============================================================================
IFS=',' read -ra IP_ARRAY <<< "$IPADDRS"

PREFILL_ARGS=""
DECODE_ARGS=""
PREFILL_PORTS=""
DECODE_PORTS=""

for ((i=1; i<=$xP && i<${#IP_ARRAY[@]}; i++)); do
    PREFILL_ARGS+="${IP_ARRAY[$i]} "
    PREFILL_PORTS+="$SERVER_PORT "
done

for ((i=xP+1; i<${#IP_ARRAY[@]}; i++)); do
    DECODE_ARGS+="${IP_ARRAY[$i]} "
    DECODE_PORTS+="$SERVER_PORT "
done

# =============================================================================
# Node Role Assignment and Server Launch
# =============================================================================

if [ "$NODE_RANK" -eq 0 ]; then
    echo "NODE INFO ======================================="
    echo "Node IPs : ${IPADDRS}"
    echo "Model    : ${MODEL_NAME} @ ${MODEL_PATH}"
    echo "Engine ID: ${ENGINE_ID}"
    echo "CLUSTER INFO ===================================="
    echo "${host_name}:${host_ip} is Proxy Node"
    echo "Prefill: ${PREFILL_ARGS}"
    echo "Decode:  ${DECODE_ARGS}"
    echo "================================================"

    PD_IPADDRS="${IPADDRS#*,}"
    echo "Waiting for all prefill and decode servers to be up . . ."
    python3 $NIXL_COOKBOOK_PATH/sync.py barrier \
        --node-ips ${PD_IPADDRS} \
        --node-ports $SERVER_PORT \
        --wait-for-all-ports \
        --timeout 1800

    if [ "$PROXY_TYPE" == "vllm_router" ]; then
        echo "Starting vLLM Router..."
        [ -f /root/.cargo/env ] && source /root/.cargo/env

        PREFILL_URLS=""
        DECODE_URLS=""
        for ip in ${PREFILL_ARGS}; do
            PREFILL_URLS+="--prefill http://${ip}:${SERVER_PORT} "
        done
        for ip in ${DECODE_ARGS}; do
            DECODE_URLS+="--decode http://${ip}:${SERVER_PORT} "
        done

        ROUTER_CMD="UCX_TLS=tcp,self,shm VLLM_USE_V1=1 \
        vllm-router \
            --host 0.0.0.0 \
            --port $ROUTER_PORT \
            --vllm-pd-disaggregation \
            $PREFILL_URLS \
            $DECODE_URLS \
            --policy round_robin \
            --prefill-policy round_robin \
            --decode-policy round_robin \
            --intra-node-data-parallel-size 1 \
            --retry-max-retries 3 \
            --health-check-endpoint /health \
            --prometheus-port 29000"

        if [[ "$DRY_RUN" -eq 1 ]]; then echo "DRY RUN: $ROUTER_CMD"
        else
            eval "$ROUTER_CMD" \
                2>&1 | tee /run_logs/${SLURM_JOB_ID}/vllm_router_NODE${NODE_RANK}.log >/dev/null &
            proxy_pid=$!
        fi
        PROXY_PORT=$ROUTER_PORT
    else
        echo "Starting Toy Proxy Server..."
        PROXY_CMD="UCX_TLS=tcp,self,shm NCCL_UCX_TLS=tcp VLLM_USE_V1=1 \
        python3 \"/app/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py\" \
                --host 0.0.0.0 \
                --port $SERVER_PORT \
                --prefiller-hosts ${PREFILL_ARGS} \
                --prefiller-ports ${PREFILL_PORTS} \
                --decoder-hosts ${DECODE_ARGS} \
                --decoder-ports ${DECODE_PORTS}"

        if [[ "$DRY_RUN" -eq 1 ]]; then echo "DRY RUN: $PROXY_CMD"
        else
            eval "$PROXY_CMD" \
                2>&1 | tee /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >/dev/null &
            proxy_pid=$!
        fi
        PROXY_PORT=$SERVER_PORT
    fi

    echo "Waiting for proxy server to be up . . ."
    python3 $NIXL_COOKBOOK_PATH/sync.py barrier \
        --node-ips ${host_ip} \
        --node-ports $PROXY_PORT \
        --wait-for-all-ports \
        --timeout 600

    echo "Proxy Server ($PROXY_TYPE) Ready on ${host_name}:${host_ip}:${PROXY_PORT}"
    sleep 10
    export BENCHMARK_PORT=$PROXY_PORT

    if [[ "$DRY_RUN" -eq 1 ]]; then echo "DRY RUN: bash $NIXL_COOKBOOK_PATH/benchmark_xPyD.sh"
    else bash $NIXL_COOKBOOK_PATH/benchmark_xPyD.sh; fi

    echo "Killing the proxy server"
    [[ "$DRY_RUN" -eq 0 ]] && kill $proxy_pid

elif [ "$NODE_RANK" -gt 0 ] && [ "$NODE_RANK" -le "$xP" ]; then
    echo "${host_name}:${host_ip} is Prefill Node (Model: ${MODEL_NAME})"
    echo "Using prefill config: $PREFILL_MODEL_CONFIG"

    # Export UCX/Nixl env vars so child processes (Workers) inherit them
    export UCX_TLS=all
    export UCX_SOCKADDR_TLS_PRIORITY=tcp
    export UCX_MEMTYPE_CACHE=y
    export UCX_RNDV_SCHEME=get_zcopy
    export UCX_RNDV_THRESH=4k
    export UCX_ROCM_IPC_MIN_ZCOPY=0
    export HSA_ENABLE_SDMA=1
    export UCX_LOG_LEVEL=info
    export NIXL_LOG_LEVEL=DEBUG
    export VLLM_USE_V1=1
    export VLLM_SERVER_DEV_MODE=0
    export VLLM_NIXL_SIDE_CHANNEL_HOST=${host_ip}
    export VLLM_NIXL_SIDE_CHANNEL_PORT=5557

    for env_pair in ${PREFILL_MODEL_ENVS}; do
        export "$env_pair"
    done

    PREFILL_CMD="vllm serve \${MODEL_PATH} \
        --port $SERVER_PORT \
        --trust-remote-code \
        --disable-log-requests \
        --kv-transfer-config '{\"kv_connector\": \"NixlConnector\", \"engine_id\": \"${ENGINE_ID}\", \"kv_role\": \"kv_producer\", \"kv_parallel_size\": 8, \"kv_rank\": 0, \"kv_buffer_size\": 5000000000, \"kv_buffer_device\": \"cuda\", \"kv_ip\": \"'\"\${rdma_ip}\"'\", \"kv_port\": 14600}'"

    if [[ -n "$PREFILL_MODEL_CONFIG" ]]; then
        PREFILL_CMD="$PREFILL_CMD $PREFILL_MODEL_CONFIG"
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then echo "DRY RUN: $PREFILL_CMD"
    else
        eval "$PREFILL_CMD" \
            2>&1 | tee /run_logs/${SLURM_JOB_ID}/prefill_NODE${NODE_RANK}.log >/dev/null &
        prefill_pid=$!
    fi

    echo "Waiting for proxy server to be up..."
    python3 $NIXL_COOKBOOK_PATH/sync.py barrier \
        --node-ips ${MASTER_ADDR} --node-ports $PROXY_PORT \
        --wait-for-all-ports --timeout 1800

    echo "Waiting until proxy server closes..."
    python3 $NIXL_COOKBOOK_PATH/sync.py wait \
        --remote-ip ${MASTER_ADDR} --remote-port $PROXY_PORT

    echo "Killing the prefill server"
    [[ "$DRY_RUN" -eq 0 ]] && kill $prefill_pid

else
    echo "${host_name}:${host_ip} is Decode Node (Model: ${MODEL_NAME})"
    echo "Using decode config: $DECODE_MODEL_CONFIG"

    export UCX_TLS=all
    export UCX_SOCKADDR_TLS_PRIORITY=tcp
    export UCX_MEMTYPE_CACHE=y
    export UCX_RNDV_SCHEME=get_zcopy
    export UCX_RNDV_THRESH=4k
    export UCX_ROCM_IPC_MIN_ZCOPY=0
    export HSA_ENABLE_SDMA=1
    export UCX_LOG_LEVEL=info
    export NIXL_LOG_LEVEL=DEBUG
    export VLLM_USE_V1=1
    export VLLM_SERVER_DEV_MODE=0
    export VLLM_NIXL_SIDE_CHANNEL_HOST=${host_ip}
    export VLLM_NIXL_SIDE_CHANNEL_PORT=5557

    for env_pair in ${DECODE_MODEL_ENVS}; do
        export "$env_pair"
    done

    DECODE_CMD="vllm serve \${MODEL_PATH} \
        --port $SERVER_PORT \
        --trust-remote-code \
        --disable-log-requests \
        --kv-transfer-config '{\"kv_connector\": \"NixlConnector\", \"engine_id\": \"${ENGINE_ID}\", \"kv_role\": \"kv_consumer\", \"kv_parallel_size\": 8, \"kv_rank\": 0, \"kv_buffer_size\": 5000000000, \"kv_buffer_device\": \"cuda\", \"kv_ip\": \"'\"\${rdma_ip}\"'\", \"kv_port\": 14600}'"

    if [[ -n "$DECODE_MODEL_CONFIG" ]]; then
        DECODE_CMD="$DECODE_CMD $DECODE_MODEL_CONFIG"
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then echo "DRY RUN: $DECODE_CMD"
    else
        eval "$DECODE_CMD" \
            2>&1 | tee /run_logs/${SLURM_JOB_ID}/decode_NODE${NODE_RANK}.log >/dev/null &
        decode_pid=$!
    fi

    echo "Waiting for proxy server to be up..."
    python3 $NIXL_COOKBOOK_PATH/sync.py barrier \
        --node-ips ${MASTER_ADDR} --node-ports $PROXY_PORT \
        --wait-for-all-ports --timeout 1800

    echo "Waiting until proxy server closes..."
    python3 $NIXL_COOKBOOK_PATH/sync.py wait \
        --remote-ip ${MASTER_ADDR} --remote-port $PROXY_PORT

    echo "Killing the decode server"
    [[ "$DRY_RUN" -eq 0 ]] && kill $decode_pid
fi

echo "Killing the etcd server"
kill $etcd_pid

echo "Script completed successfully"
exit 0
