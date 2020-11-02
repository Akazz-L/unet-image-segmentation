ARG UBUNTU_VERSION=18.04
ARG CUDA=10.1

FROM nvidia/cuda:${CUDA}-base-ubuntu${UBUNTU_VERSION}
ARG CUDA

ENV CUDNN=7.6.5.32-1
ARG CUDNN_MAJOR_VERSION=7
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=6.0.1-1
ARG LIBNVINFER_MAJOR_VERSION=6

ENV HOME /root
ENV PYTHON_VERSION 3.7.4
ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends apt-utils

RUN apt-get update && apt-get install -y --no-install-recommends --allow-unauthenticated \
    build-essential \
    ca-certificates \
    cuda-command-line-tools-${CUDA/./-} \
    libcublas10=10.2.1.243-1 \
    cuda-cudart-${CUDA/./-} \
    cuda-nvrtc-${CUDA/./-} \
    cuda-cufft-${CUDA/./-} \
    cuda-curand-${CUDA/./-} \
    cuda-cusolver-${CUDA/./-} \
    cuda-cusparse-${CUDA/./-} \
    libcudnn7=${CUDNN}+cuda${CUDA} \
    # TensorFlow doesn't require libnccl anymore but Open MPI still depends on it
    libnccl2=2.4.7-1+cuda10.1 \
    libgomp1 \
    libnccl-dev=2.4.7-1+cuda10.1 \
    libfreetype6-dev \
    libhdf5-serial-dev \
    liblzma-dev \
    libpng-dev \
    libtemplate-perl \
    libzmq3-dev \
    curl \
    git \
    emacs \
    wget \
    vim \
    libffi-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    python-openssl \
    openssh-client \
    openssh-server \
    zlib1g-dev \
    # Install dependent library for OpenCV
    libgtk2.0-dev \
    pkg-config \
    software-properties-common \
    unzip

# Install TensorRT if not building for PowerPC
RUN [[ "${ARCH}" = "ppc64le" ]] || { apt-get update && \
    apt-get install -y --no-install-recommends libnvinfer${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
    libnvinfer-plugin${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*; }

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV LD_INCLUDE_PATH=/usr/local/cuda/include:/usr/local/cuda/extras/CUPTI/include:$LD_INCLUDE_PATH

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

# Install Python
RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN pyenv install $PYTHON_VERSION
RUN pyenv global $PYTHON_VERSION

# Install Python project dependencies
RUN pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel

RUN pip install --no-cache-dir --upgrade sagemaker-training 

# code and data files
ENV CODE_PATH=/opt/ml/code
ENV MODEL_PATH=/opt/ml/model
ENV INPUT_PATH=/opt/ml/input

COPY . $CODE_PATH
RUN pip install -r /opt/ml/code/requirements.txt


RUN ln -s $INPUT_PATH/data $CODE_PATH/data
RUN ln -s $MODEL_PATH $CODE_PATH/output

ENV PYTHONPATH=$CODE_PATH
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

ENV SAGEMAKER_PROGRAM train.py