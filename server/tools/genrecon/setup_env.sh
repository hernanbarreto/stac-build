#!/bin/bash
# GenRecon environment (the README's "manual" path: the CUDA toolkit inside the env, torch cu126)
set -x
export MAX_JOBS=8 OMP_NUM_THREADS=8 TORCH_CUDA_ARCH_LIST="8.0"
CONDA=/workspace/miniforge3/bin/conda
source /workspace/miniforge3/etc/profile.d/conda.sh
$CONDA create -y -n genrecon python=3.10 nvidia::cuda-toolkit=12.6 ninja || exit 1
conda activate genrecon
export CUDA_HOME=$CONDA_PREFIX
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126 || exit 1
conda install -y -c conda-forge libjpeg-turbo xorg-libx11
pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers==4.57.3 gradio==6.0.1 tensorboard pandas lpips zstandard
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
pip install pillow-simd || echo "pillow-simd skipped"
pip install kornia timm
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl || pip install flash-attn==2.7.3 --no-build-isolation
mkdir -p /tmp/claude-0/genrecon_ext && cd /tmp/claude-0/genrecon_ext
[ -d nvdiffrast ] || git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git nvdiffrast
pip install ./nvdiffrast --no-build-isolation
[ -d nvdiffrec ] || git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git nvdiffrec
pip install ./nvdiffrec --no-build-isolation
[ -d CuMesh ] || git clone https://github.com/JeffreyXiang/CuMesh.git CuMesh --recursive
pip install ./CuMesh --no-build-isolation
[ -d FlexGEMM ] || git clone https://github.com/JeffreyXiang/FlexGEMM.git FlexGEMM --recursive
pip install ./FlexGEMM --no-build-isolation
cd /workspace/stac-build/vendor/GenRecon && git submodule update --init --recursive
rm -rf /tmp/claude-0/genrecon_ext/o-voxel && cp -r o-voxel /tmp/claude-0/genrecon_ext/o-voxel && pip install /tmp/claude-0/genrecon_ext/o-voxel --no-build-isolation
python -c "import torch, flash_attn, nvdiffrast, cumesh, flex_gemm, o_voxel; print('OK torch', torch.__version__, torch.cuda.is_available())" 2>&1 | tail -3
echo "SETUP DONE"
