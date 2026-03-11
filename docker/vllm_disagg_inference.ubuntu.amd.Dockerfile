ARG BASE_IMAGE=vllm/vllm-openai-rocm:v0.17.1
FROM ${BASE_IMAGE}

WORKDIR /root

ENV ROCM_PATH=/opt/rocm

# ---------------------------------------------------------------------------
# UCX build (ROCm fork, matching upstream vLLM Dockerfile.rocm build_rixl)
# ---------------------------------------------------------------------------
ENV UCX_HOME=/usr/local/ucx

RUN apt-get update -q -y && apt-get install -q -y \
    autoconf automake libtool pkg-config \
    librdmacm-dev rdmacm-utils libibverbs-dev ibverbs-utils ibverbs-providers \
    infiniband-diags perftest ethtool rdma-core strace \
    && rm -rf /var/lib/apt/lists/*

ARG UCX_REPO=https://github.com/ROCm/ucx.git
ARG UCX_BRANCH=da3fac2a
RUN cd /usr/local/src && \
    git clone ${UCX_REPO} && cd ucx && \
    git checkout ${UCX_BRANCH} && \
    ./autogen.sh && mkdir build && cd build && \
    ../configure \
        --prefix=${UCX_HOME} \
        --enable-shared --disable-static \
        --disable-doxygen-doc --enable-optimizations \
        --enable-devel-headers --enable-mt \
        --with-rocm=${ROCM_PATH} --with-verbs --with-dm && \
    make -j$(nproc) && make install && \
    rm -rf /usr/local/src/ucx

ENV PATH=${UCX_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${UCX_HOME}/lib:${LD_LIBRARY_PATH}

# ---------------------------------------------------------------------------
# RIXL / Nixl build (matching upstream vLLM Dockerfile.rocm build_rixl)
# ---------------------------------------------------------------------------
ENV RIXL_HOME=/usr/local/rixl

RUN apt-get update -q -y && apt-get install -q -y \
    libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc \
    libcpprest-dev libaio-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install meson pybind11[global]

ARG RIXL_REPO=https://github.com/ROCm/RIXL.git
ARG RIXL_BRANCH=f33a5599
RUN git clone ${RIXL_REPO} /opt/rixl && cd /opt/rixl && \
    git checkout ${RIXL_BRANCH} && \
    meson setup build --prefix=${RIXL_HOME} \
        -Ducx_path=${UCX_HOME} \
        -Drocm_path=${ROCM_PATH} && \
    cd build && ninja && ninja install

RUN cd /opt/rixl && \
    pip install \
        --config-settings=setup-args="-Drocm_path=${ROCM_PATH}" \
        --config-settings=setup-args="-Ducx_path=${UCX_HOME}" . && \
    rm -rf /opt/rixl

ENV LD_LIBRARY_PATH=${RIXL_HOME}/lib:${RIXL_HOME}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# ---------------------------------------------------------------------------
# etcd (required for vLLM disagg service discovery)
# ---------------------------------------------------------------------------
ARG ETCD_VERSION=v3.6.0-rc.5
RUN wget -q https://github.com/etcd-io/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-linux-amd64.tar.gz -O /tmp/etcd.tar.gz && \
    mkdir -p /usr/local/bin/etcd && \
    tar -xf /tmp/etcd.tar.gz -C /usr/local/bin/etcd --strip-components=1 && \
    rm /tmp/etcd.tar.gz
ENV PATH=${PATH}:/usr/local/bin/etcd

# ---------------------------------------------------------------------------
# AMD Pensando ionic RDMA verbs provider (for RoCEv2 KV transfer via Nixl)
# ---------------------------------------------------------------------------
COPY libionic1_54.0-149.g3304be71_amd64.deb /tmp/libionic1.deb
RUN dpkg -i /tmp/libionic1.deb && rm /tmp/libionic1.deb

# ---------------------------------------------------------------------------
# vllm-router (Rust-based proxy for PD disaggregation)
# ---------------------------------------------------------------------------
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install vllm-router

ENTRYPOINT []
CMD ["/bin/bash"]
