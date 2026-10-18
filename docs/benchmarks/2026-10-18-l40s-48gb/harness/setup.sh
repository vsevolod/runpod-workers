#!/bin/bash
set -euo pipefail
export UV_CACHE_DIR=/opt/cache/uv
export UV_LINK_MODE=copy
mkdir -p /opt/bench /opt/venvs /models
git clone https://github.com/vsevolod/runpod-workers /opt/runpod-workers
git -C /opt/runpod-workers checkout 6f8cf4fcfba8da763348156736530376af02c715
python3 -m pip install --no-cache-dir uv
uv venv --system-site-packages /opt/venvs/krea
printf '%s\n' 'torch==2.8.0+cu128' 'torchvision==0.23.0+cu128' > /opt/bench/torch-constraints.txt
uv pip install --python /opt/venvs/krea/bin/python 'torch==2.8.0+cu128' 'torchvision==0.23.0+cu128' --index https://download.pytorch.org/whl/cu128
uv pip install --python /opt/venvs/krea/bin/python --no-cache -c /opt/bench/torch-constraints.txt -r /opt/runpod-workers/workers/krea2/requirements.txt
uv venv --system-site-packages /opt/venvs/qwen
uv pip install --python /opt/venvs/qwen/bin/python 'torch==2.8.0+cu128' 'torchvision==0.23.0+cu128' --index https://download.pytorch.org/whl/cu128
uv pip install --python /opt/venvs/qwen/bin/python --no-cache -c /opt/bench/torch-constraints.txt -r /opt/runpod-workers/workers/qwen_image_21/requirements.txt
/opt/venvs/qwen/bin/python - <<'PY'
import subprocess, sysconfig
subprocess.run(['git', 'apply', '--check', '/opt/runpod-workers/workers/qwen_image_21/patches/diffusers-native-references.patch'], cwd=sysconfig.get_path('purelib'), check=True)
subprocess.run(['git', 'apply', '/opt/runpod-workers/workers/qwen_image_21/patches/diffusers-native-references.patch'], cwd=sysconfig.get_path('purelib'), check=True)
PY
uv pip freeze --python /opt/venvs/krea/bin/python > /opt/bench/krea-freeze.txt
uv pip freeze --python /opt/venvs/qwen/bin/python > /opt/bench/qwen-freeze.txt
nvidia-smi -q > /opt/bench/nvidia-smi.txt
echo SETUP_COMPLETE
