# Base image
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y locales && rm -rf /var/lib/apt/lists/* \
	&& localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8

# install system related packages
RUN apt-get update && \
    apt-get install -y \
    cmake \
    make \
    nano \
    vim \
    curl \
    git \
    pip \
    software-properties-common \
    build-essential \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libzmq3-dev \
    '^libxcb.*-dev' \
    libx11-dev \
    libx11-xcb-dev \
    libxi-dev \
    libxrender-dev \
    libxkbcommon-dev \
    libxkbcommon-x11-dev \
    libglu1-mesa-dev \
    python3-tk \
    && rm -rf /var/lib/apt/lists/* \
    libglfw3-dev


# Create a non-root user
# Linux default main user with UID 1000
# USERNAME is not important. files created will still be accessable 
ARG USERNAME=atari
ARG USER_UID=1000 
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
  && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && mkdir /home/$USERNAME/.config && chown $USER_UID:$USER_GID /home/$USERNAME/.config


# Set up sudo
RUN apt-get update \
  && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  && rm -rf /var/lib/apt/lists/*

# dependencies
RUN sudo add-apt-repository main -y\
  && sudo apt-get update \
  && sudo apt-get install -y \
  coinor-libipopt-dev \
  libeigen3-dev \
  ffmpeg\
  texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super -y\
  && rm -rf /var/lib/apt/lists/*

# Pybind11 global
RUN pip install pybind11[global]

# Dependencies
RUN python3 -m pip install --upgrade pip pytest matplotlib jupyter PyQt5
RUN python3 -m pip install --upgrade robot_descriptions
RUN python3 -m pip install --upgrade tyro torch diffusers tensorboard scikit-learn einops
RUN python3 -m pip install --upgrade opencv-python xacro moviepy keyboard
# Debug Dependencies
RUN pip3 install memray memory-profiler Pympler


# Install Bullet utils + Robot properties solo
RUN git clone https://github.com/machines-in-motion/bullet_utils
RUN cd bullet_utils && python3 -m pip install .
RUN git clone https://github.com/open-dynamic-robot-initiative/robot_properties_solo
RUN cd robot_properties_solo && python3 -m pip install . --upgrade --force-reinstall --no-cache  -e ../bullet_utils/

# Install morpho sym
# RUN git clone https://github.com/Danfoa/MorphoSymm.git
# RUN cd MorphoSymm && python3 -m pip install -e .

# Install robotpkg
RUN sudo mkdir -p /etc/apt/keyrings
RUN curl http://robotpkg.openrobots.org/packages/debian/robotpkg.asc | sudo tee /etc/apt/keyrings/robotpkg.asc
RUN sudo tee /etc/apt/sources.list.d/robotpkg.list <<EOF
deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub jammy robotpkg
EOF
RUN sudo apt-get update

# manually install crocoddyl 1.9.0 and its dependencies
RUN sudo apt-get update \
  && sudo apt-get install -y \
  robotpkg-py310-eigenpy=2.7.11 \
  robotpkg-hpp-fcl=1.8.1 robotpkg-py310-hpp-fcl=1.8.1 \
  robotpkg-pinocchio=2.6.9 robotpkg-py310-pinocchio=2.6.9 \
  robotpkg-example-robot-data=4.0.1 robotpkg-py310-example-robot-data=4.0.1 \
  robotpkg-py310-crocoddyl=1.9.0 \
  && rm -rf /var/lib/apt/lists/*

RUN echo 'export PATH=/opt/openrobots/bin:$PATH' >> /root/.bashrc
RUN echo 'export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH' >> /root/.bashrc
RUN echo 'export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH' >> /root/.bashrc
RUN echo 'export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH' >> /root/.bashrc
RUN echo 'export PYTHONPATH=/opt/openrobots/lib/python3.10/site-packages:$PYTHONPATH' >> /root/.bashrc   # Watchout for Python version!! 
RUN echo 'export export CPLUS_INCLUDE_PATH=/opt/openrobots/include/:$CPLUS_INCLUDE_PATH' >> /root/.bashrc

SHELL ["/bin/bash", "-c"]
RUN echo 'export PS1="\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\[\033[01;37m\]\u\[\033[00m\]@\[\033[01;32m\]\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\] "'  >> /root/.bashrc
RUN source /root/.bashrc

# Install BiConMP
RUN git clone --recurse-submodules -j12 https://github.com/machines-in-motion/biconvex_mpc.git

COPY biconmpc_fix.py .
RUN python3 biconmpc_fix.py
RUN rm biconmpc_fix.py

RUN cd biconvex_mpc && mkdir -p build
RUN source /root/.bashrc && cd biconvex_mpc/build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/openrobots
RUN cd biconvex_mpc/build && make install -j16

RUN echo 'export PYTHONPATH=/biconvex_mpc/build:$PYTHONPATH' >> /root/.bashrc

# Downgrade numpy
RUN python3 -m pip install "numpy<2.0.0"

ENV DEBIAN_FRONTEND=noninteractive
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
 
# Good practice
USER root

# Set the working directory and source the bashrc file
RUN source /root/.bashrc
RUN cat /root/.bashrc >> /home/$USERNAME/.bashrc

CMD ["bash"]