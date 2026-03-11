# vLLM Disaggregated Prefill-Decode (P/D) Benchmark

Runs [vLLM](https://github.com/vllm-project/vllm) with **prefill-decode disaggregation** across a multi-node Slurm cluster. Prefill and decode stages execute on separate nodes, with KV-cache transferred via [RIXL/Nixl](https://github.com/ROCm/RIXL) over RDMA, and requests routed through vllm-router.

## Architecture

```
                       ┌───────────────────────────────┐
                       │       Slurm Job (sbatch)      │
                       │    run_xPyD_models.slurm      │
                       └───────────────┬───────────────┘
                  srun launches Docker containers on each node
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
  ┌───────▼──────────┐    ┌────────────▼────────────┐    ┌──────────▼───────────┐
  │ NODE 0 (Proxy)   │    │ NODE 1..xP (Prefill)    │    │ NODE xP+1..  (Dec)   │
  │                  │    │                         │    │                      │
  │  etcd server     │    │  etcd server            │    │  etcd server         │
  │  vllm-router     │    │  vllm serve             │    │  vllm serve          │
  │  benchmark       │    │    (kv_producer)        │    │    (kv_consumer)     │
  │                  │    │                         │    │                      │
  └───────┬──────────┘    └────────────┬────────────┘    └──────────┬───────────┘
          │           ◄── KV cache transfer (Nixl/RDMA) ──►         │
          │                            │                            │
          └──── HTTP (management IP)  ─┴────────────────────────────┘
```

**Total nodes required = xP + yD + 1** (the +1 is the proxy/router node).

## Prerequisites

- **Slurm cluster** with at least `xP + yD + 1` nodes (e.g., 4 nodes for 1P+2D)
- **AMD Instinct MI355X GPUs** (8 per node), ROCm drivers installed
- **RDMA-capable NICs** (e.g., Pensando ionic, Mellanox)
- **NFS-shared storage** for model weights and scripts
- **Docker** installed on all nodes (`sudo` access for remote nodes)
- **Model weights** on NFS (e.g., `/nfsdata/hf_hub_cache-0/`)

## Supported Models

| Model | TP Size | Notes |
|---|---|---|
| `DeepSeek-R1-0528` | 8 | Piecewise cudagraph, block-size 1 |
| `DeepSeek-V3` | 8 | Piecewise cudagraph, block-size 1 |
| `Llama-3.1-405B-Instruct-FP8-KV` | 8 | FP8 KV cache |
| `amd-Llama-3.3-70B-Instruct-FP8-KV` | 8 | FP8 KV cache, 64K context |
| `gpt-oss-120b` | 8 | |

Model-specific vLLM flags and environment variables are defined in `vllm_disagg_server.sh`.

## Quick Start

### Step 1: Build the Docker Image

On the head node:

```bash
cd ~/MAD/docker
docker build -t vllm_disagg_pd:latest -f vllm_disagg_inference.ubuntu.amd.Dockerfile .
```

This builds on `rocm/vllm:v0.14.0_amd_dev` and installs:
- UCX v1.19.x (ROCm fork, with ROCm RDMA support)
- RIXL/Nixl (KV-cache transfer library)
- etcd v3.6.0-rc.5 (distributed coordination)
- Rust toolchain + vllm-router (request routing)

Build takes approximately 20-30 minutes.

### Step 2: Distribute the Image

Save to NFS and load on all worker nodes:

```bash
docker save vllm_disagg_pd:latest | pigz > /nfsdata/vllm_disagg_pd_latest.tar.gz

for node in GPU74C0 GPU7418 GPU3E76; do
    ssh $node "pigz -dc /nfsdata/vllm_disagg_pd_latest.tar.gz | sudo docker load" &
done
wait
```

Use `sudo docker load` if the remote user doesn't have direct Docker permissions.

### Step 3: Deploy Scripts to NFS

The Slurm job expects scripts on a shared filesystem accessible from all nodes:

```bash
mkdir -p /nfsdata/vllm_disagg_scripts
cp ~/MAD/scripts/vllm_dissag/* /nfsdata/vllm_disagg_scripts/
```

The NFS path is configurable via `NIXL_REPO_DIR` (default: `/nfsdata/vllm_disagg_scripts`).

### Step 4: Submit the Job

```bash
cd /nfsdata/vllm_disagg_scripts

export xP=1                  # number of prefill nodes
export yD=2                  # number of decode nodes
export MODEL_NAME=DeepSeek-R1-0528

sbatch -N 4 -n 4 \
  --nodelist=GPU3D78,GPU3E76,GPU74C0,GPU7418 \
  run_xPyD_models.slurm
```

> **Note on node ordering:** Slurm sorts the nodelist alphabetically via
> `scontrol show hostnames`. The first node in sorted order becomes NODE 0
> (proxy/router). Plan accordingly when choosing nodes.

> **Note on DRY_RUN:** If you previously ran `export DRY_RUN=1` for
> debugging, make sure to `unset DRY_RUN` or `export DRY_RUN=0` before
> submitting a real job.

### Step 5: Monitor Progress

**Slurm output** (on the submission node):

```bash
squeue -u $(whoami)
tail -f /tmp/vllm-pd-slurm-<JOB_ID>.out
```

**Per-node logs** are written to `/tmp/vllm_disagg_logs/<JOB_ID>/` on each
host. Since `/tmp` is node-local, you must SSH to the specific node:

```bash
# Router + benchmark logs (on NODE 0, the alphabetically-first node)
ssh GPU3D78 "tail -f /tmp/vllm_disagg_logs/<JOB_ID>/vllm_router_NODE0.log"
ssh GPU3D78 "tail -f /tmp/vllm_disagg_logs/<JOB_ID>/benchmark_*.log"

# Prefill log (on the prefill node)
ssh GPU3E76 "tail -f /tmp/vllm_disagg_logs/<JOB_ID>/prefill_NODE1.log"

# Decode logs (on decode nodes)
ssh GPU74C0 "tail -f /tmp/vllm_disagg_logs/<JOB_ID>/decode_NODE2.log"
ssh GPU7418 "tail -f /tmp/vllm_disagg_logs/<JOB_ID>/decode_NODE3.log"
```

If the sbatch submission node is also NODE 0, router/benchmark logs are
accessible locally without SSH.

## Environment Variables

All variables have sensible defaults and are optional unless noted.

| Variable | Default | Description |
|---|---|---|
| `xP` | `1` | Number of prefill nodes |
| `yD` | `1` | Number of decode nodes |
| `MODEL_NAME` | *(required)* | One of the supported model names |
| `DOCKER_IMAGE_NAME` | `vllm_disagg_pd:latest` | Docker image to use |
| `PROXY_TYPE` | `vllm_router` | `vllm_router` or `toy_proxy` |
| `ROUTER_PORT` | `2584` | Port for the router and vLLM servers |
| `MODEL_DIR` | `/nfsdata/hf_hub_cache-0` | Base directory for model weight search |
| `DRY_RUN` | `0` | Set to `1` to print commands without executing |
| `LOG_PATH` | `/tmp/vllm_disagg_logs` | Host-side log directory |
| `NIXL_REPO_DIR` | `/nfsdata/vllm_disagg_scripts` | NFS path where scripts are deployed |
| `BENCH_NUM_PROMPTS_MULTIPLIER` | `10` | `num_prompts = concurrency * multiplier` |

## Benchmark Configuration

The benchmark (`benchmark_xPyD.sh`) runs a concurrency sweep with `vllm bench serve`:

- **Input/output length combinations:** 1024/1024, 8192/1024, 1024/8192
- **Concurrency levels:** 8, 16, 32, 64, 128, 256, 512
- **Request rate:** infinite (max throughput)
- **Total runs:** 3 combinations x 7 concurrency levels = 21 benchmark points

Results are written to
`/tmp/vllm_disagg_logs/<JOB_ID>/benchmark_<JOB_ID>_<timestamp>_*_CONCURRENCY.log`
on the proxy node.

## Networking

The scripts distinguish between two network planes:

- **Management network** (e.g., `172.18.170.x`): Used for Slurm communication,
  etcd cluster formation, vllm-router proxying, barrier synchronization, and
  benchmark traffic. Resolved via `ip route get 1.1.1.1`.

- **RDMA network** (e.g., `192.168.1.x`): Used for Nixl KV-cache transfers
  between prefill and decode nodes. Resolved via `hostname -I | grep 192.168`.
  Falls back to the management IP if no RDMA IP is found.

RDMA device detection is handled by `env.sh`, which auto-detects IB devices
based on hostname patterns (e.g., `GPU*` -> Pensando ionic NICs) or
`ibv_devinfo` output.

## File Descriptions

| File | Purpose |
|---|---|
| `run_xPyD_models.slurm` | Slurm batch script: resolves IPs, finds model weights, launches Docker containers |
| `vllm_disagg_server.sh` | Per-container entrypoint: assigns node roles, starts etcd/vllm/router/benchmark |
| `env.sh` | Auto-detects RDMA devices and sets UCX/NCCL environment variables |
| `start_etcd.sh` | Starts a local etcd node and joins the cluster |
| `sync.py` | TCP barrier and shutdown coordination between nodes |
| `benchmark_xPyD.sh` | Runs the concurrency sweep benchmark against the router |

## Known Issues and Workarounds

### Nixl UCX "no active messages transport" error

**Symptom:** Prefill/decode servers crash with:
```
UCX ERROR no active messages transport to <no debug data>:
  self/memory - no peer failure handler, ...
nixl_agent.cpp: createBackend: backend 'UCX' encountered error during
  intra-agent transfer setup with status NIXL_ERR_BACKEND
```

**Cause:** The Nixl UCX backend defaults to `UCP_ERR_HANDLING_MODE_PEER`, which
requires peer-failure-handler support. Shared-memory transports (self, sysv,
posix) don't provide this, and Pensando ionic NICs lack RDMA CM support for
reliable-connected transports.

**Fix:** `vllm_disagg_server.sh` automatically patches the RIXL Python wrapper
at container startup to set `ucx_error_handling_mode=none`. This is a no-op if
the patch has already been applied or a newer RIXL version fixes it upstream.

### Docker permissions on remote nodes

The Slurm script uses `sudo docker` for all container operations. Ensure the
Slurm user has passwordless `sudo` for Docker commands on all worker nodes.

### Model path resolution

The script searches multiple NFS mount points for model weights:
1. `/nfsdata/hf_hub_cache-0/<model-dir>/snapshots/<hash>`
2. `/mnt/m2m_nobackup/models_blog/<model-dir>`
3. `/shared_inference/models_blog/<model-dir>`
4. `$MODEL_DIR/<model-dir>`

For HuggingFace cache layouts, it automatically resolves the snapshot hash
directory. Set `MODEL_DIR` to override the base search path.
