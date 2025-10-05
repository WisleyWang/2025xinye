
# FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
RUN sed -i 's/https:\/\/mirrors.aliyun.com/http:\/\/mirrors.cloud.aliyuncs.com/g' /etc/apt/sources.list

#设置时区
ENV TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive
    

# 修改shell为bash
RUN rm /bin/sh && ln -s /usr/bin/bash /bin/sh

# Install dependencies outside of the base image
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
	apt-get install -y --no-install-recommends automake \
    build-essential  \
    ca-certificates  \
    libfreetype6-dev  \
    libtool  \
    pkg-config  \
    ffmpeg \
    tmux \
    vim
 


# 设置环境变量 url
ENV HOME /algorithm
ENV HF_ENDPOINT https://hf-mirror.com \
    HF_HUB_DISABLE_PROGRESS_BARS=true


# 拷贝代码到容器中
COPY . ${HOME}/
# COPY ./requirements.txt ${HOME}/requirements.txt
# 切换工作空间到对应目录
WORKDIR ${HOME}   
# pip 更换为清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 安装依赖
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt


# 清理 APT 缓存
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -Rf /tmp/*
# 清理pip缓存
RUN rm -rf ${HOME}/.cache/pip
# 清理 CONDA 缓存
RUN find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    conda clean -afy