#!/bin/bash
# OpenMVS (cdcseacave/openMVS master, AGPL-3.0) with CUDA for the 2026-10-17 trial against GenRecon — the whole recipe,
# idempotent. The env `openmvs` holds the conda-forge parts; master also needs six libraries conda-forge does not ship
# (or ships too old), built from source into the same prefix; OpenMVS's own vcpkg ports pin the forks/commits used here.
#   bash setup_env.sh            (≈ 25 min: env 6 GB, six small libs, OpenMVS ~15 min on 14 cores)
set -e
M=/workspace/miniforge3/bin/mamba; E=/workspace/miniforge3/envs/openmvs; V=/workspace/stac-build/vendor; T=$(cd "$(dirname "$0")" && pwd)
export PATH=$E/bin:/usr/local/cuda/bin:$PATH
if [ ! -x $E/bin/cmake ]; then
  # Eigen is NOT pinned: conda's ceres/poselib builds resolve it (5.0.1 on 2026-10-17); OpenCV 4 headless (5.0 is what
  # the solver picks otherwise); ceres as the CPU build (the GPU build demands a CUDA toolkit newer than the pod's 12.8)
  $M create -y -n openmvs -c conda-forge cmake ninja "libopencv=4.*=headless*" cgal-cpp libboost-devel gmp mpfr \
     nanoflann "ceres-solver=*=cpu*" tinyxml2 libheif libjxl tinygltf suitesparse fast_float
fi
clone() { local dir=$1 url=$2 ref=$3; if [ ! -d $V/$dir ]; then mkdir -p $V/$dir && git -C $V/$dir init -q && git -C $V/$dir remote add origin $url \
          && git -C $V/$dir fetch -q --depth 1 origin $ref && git -C $V/$dir checkout -q FETCH_HEAD; fi; }
build() { local dir=$1; shift; rm -rf $V/$dir/_build && mkdir -p $V/$dir/_build && (cd $V/$dir/_build && cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=$E -DCMAKE_PREFIX_PATH=$E -DEigen3_DIR=$E/share/eigen3/cmake -DOpenCV_DIR=$E/lib/cmake/opencv4 "$@" > cmake.log 2>&1 \
          && nice -n 10 ninja -j12 > ninja.log 2>&1 && ninja install > install.log 2>&1) && echo "$dir installed"; }
clone vcglib      https://github.com/cdcseacave/VCG.git          658ba36       # headers only (GPL-3.0), OpenMVS's own VCG fork
clone TinyEXIF    https://github.com/cdcseacave/TinyEXIF.git     5445d04007d9d0f5296f12aa4fc637ea68b8368f   # MIT
clone TinyNPY     https://github.com/cdcseacave/TinyNPY.git      d36d3c3ed4372a30061676ad5715937af736aa32   # MIT
clone tinyply     https://github.com/ddiakopoulos/tinyply.git    c9bb690dfe5e9105961e9e28120c48c9ae084bc6   # BSD-2 / public domain
clone thread-pool https://github.com/bshoshany/thread-pool.git   bd4533f1f70c2b975cbd5769a60d8eaaea1d2233   # MIT, one header
clone halfmesh    https://github.com/cdcseacave/halfmesh.git     f218f7c1a41218cc71990bfec0b1557a2c1b60e8   # MIT, v0.5.1 = OpenMVS's port
clone PoseLib     https://github.com/cdcseacave/PoseLib.git      ccdd2f62d7ee91b41a1dce4dfd619b688b6c247a   # BSD-3; the FORK OpenMVS's port pins (bearings API)
clone openMVS     https://github.com/cdcseacave/openMVS.git      610562a
[ -f $E/lib/cmake/TinyEXIF/TinyEXIFConfig.cmake ] || build TinyEXIF -DBUILD_DEMO=OFF -DBUILD_SHARED_LIBS=ON
[ -f $E/lib/cmake/TinyNPY/TinyNPYConfig.cmake ]   || build TinyNPY -DBUILD_DEMO=OFF
[ -d $E/lib/cmake/tinyply ]                       || build tinyply -DSHARED_LIB=ON -DBUILD_TESTS=OFF
[ -f $E/include/BS_thread_pool.hpp ]              || cp $V/thread-pool/include/BS_thread_pool.hpp $E/include/
[ -d $E/lib/cmake/halfmesh ]                      || build halfmesh -DBUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON   # static without -fPIC cannot enter libMVS.so (TLS relocation)
[ -d $E/lib/cmake/PoseLib ]                       || build PoseLib -DMARCH_NATIVE=OFF -DWITH_BENCHMARK=OFF -DBUILD_TESTS=OFF -DPYTHON_PACKAGE=OFF -DWERROR=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
# OpenMVS itself: its repo's build/ holds its cmake modules, so the binaries go to _build/; ceres's FindSuiteSparse asks
# for OpenMP's C component and OpenMVS enables only CXX → enable_c.cmake is injected after project()
mkdir -p $V/openMVS/_build && cd $V/openMVS/_build
export CUDAHOSTCXX=$(which g++)
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$E -DBOOST_ROOT=$E -DEigen3_DIR=$E/share/eigen3/cmake -DOpenCV_DIR=$E/lib/cmake/opencv4 \
  -DCGAL_DIR=$E/lib/cmake/CGAL -DVCG_ROOT=$V/vcglib -DOpenMVS_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_PROJECT_OpenMVS_INCLUDE=$T/enable_c.cmake -DOpenMVS_USE_PYTHON=OFF -DOpenMVS_USE_BREAKPAD=OFF -DOpenMVS_ENABLE_TESTS=OFF -DOpenMVS_USE_CERES=OFF \
  -DCMAKE_INSTALL_PREFIX=$V/openMVS/install > cmake.log 2>&1
nice -n 10 ninja -j14 2>&1 | grep -vE "^\[[0-9]+/[0-9]+\] Building" | tail -40
ls $V/openMVS/_build/bin/
