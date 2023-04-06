FROM nvcr.io/nvidia/pytorch:23.02-py3

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai LANG=C.UTF-8 LC_ALL=C.UTF-8 PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONHASHSEED=0

RUN \
    # sed -i "s/archive.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list &&\
    # sed -i "s/security.ubuntu.com/mirrors.ustc.edu.cn/g" /etc/apt/sources.list &&\
    rm -f /etc/apt/sources.list.d/* &&\
    apt-get update && apt-get upgrade -y &&\
    apt-get install -y --no-install-recommends \
        autoconf automake autotools-dev build-essential ca-certificates \
        make cmake ninja-build pkg-config g++ ccache yasm \
        git curl wget unzip nano net-tools htop iotop \
        cloc rsync xz-utils software-properties-common tzdata \
    && apt-get purge -y unattended-upgrades \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN \
    # pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple &&\
    pip install -U pip &&\
    pip install "scikit-learn<1" scikit-image configargparse "imageio[ffmpeg]" "jupyterlab~=3.5.0" "jupyter-archive~=3.2"
