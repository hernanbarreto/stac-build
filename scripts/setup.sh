#!/bin/bash
set -e

echo "🚀 STAC-BUILD Environment Setup"
echo "=============================="

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "⚠️  Conda not found! Please install Miniconda or Anaconda."
    exit 1
fi

# Create Environment from YAML
echo "📦 Creating/Updating Conda Environment 'stac-build'..."
conda env update -f environment.yml --prune

# Activate environment to verify
# Note: In bash scripts, activating conda can be tricky. We use source.
source $(conda info --base)/etc/profile.d/conda.sh
conda activate stac-build

echo "✅ Environment 'stac-build' ready."

# Install PyTorch3D or other heavy deps if needed manually
# (Added to environment.yml via pip section where possible)
# Avoid local compilation which fails due to driver/toolkit mismatch.
# Use pre-built wheel for CUDA 11.8 (compatible with our environment or newer drivers)
# Note: We are using a specific version known to work.
echo "� Installing gsplat (pre-built wheel)..."
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt21cu121

echo "🔗 Linking local repositories..."
# This is crucial for the Monolithic architecture
# We add .pth files to the site-packages so python finds the modules

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
USER_HOME="/home/hernan"

echo "   📍 Site Packages: $SITE_PACKAGES"

# Link Depth-Anything-3
echo "$USER_HOME/Depth-Anything-3/src" > "$SITE_PACKAGES/da3.pth"
echo "   - Linked Depth-Anything-3"

# Link SAM3
echo "$USER_HOME/sam3" > "$SITE_PACKAGES/sam3.pth"
echo "   - Linked SAM3"

echo "🎉 Setup Complete! Run './scripts/start.sh' to launch."
