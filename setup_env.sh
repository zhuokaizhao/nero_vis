#!/bin/bash

# check if NVidia GPU is available
if [[ $(lshw -C display | grep vendor) =~ "NVIDIA Corporation" ]]; then
    echo "Installing PyTorch with CUDA"
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
else
    echo "Installing CPU-only PyTorch"
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
fi

# other dependencies
pip install cupy
pip install flowiz
pip install PySide6
pip install pyqtgraph
pip install opencv-python
conda install -c conda-forge tqdm -y
conda install -c conda-forge pillow -y
conda install -c conda-forge argparse -y
conda install -c conda-forge scikit-learn -y
conda install -c conda-forge umap-learn -y
conda install -c conda-forge einops -y
conda install -c conda-forge hydra-core -y
conda install -c conda-forge omegaconf -y
pip install e2cnn