#!/usr/bin/env bash
set -euo pipefail

# JupyterLab 터미널/SSH에서 실행 가능한 GPU 클러스터용 helper script.
# 기본 동작: venv 생성 -> deps 설치 -> (옵션) 데이터 clone -> dataset build -> baseline/ViT train -> eval/report
#
# 예시:
#   bash scripts/run_vit_cluster.sh --mode smoke --repo-root "$PWD"
#   bash scripts/run_vit_cluster.sh --mode train --config configs/vit_mid.yaml
#   bash scripts/run_vit_cluster.sh --mode full --config configs/vit_gpu_full.yaml --clone-data
#
# 환경변수(선택):
#   PYTHON_BIN=python3.10
#   CUDA_VISIBLE_DEVICES=0
#   HF_HOME=/path/to/cache
#   TORCH_HOME=/path/to/cache
#   XDG_CACHE_HOME=/path/to/cache
#   MPLCONFIGDIR=/path/to/cache

MODE="smoke"
CONFIG="configs/vit_mid.yaml"
REPO_ROOT="$(pwd)"
CLONE_DATA=0
SKIP_INSTALL=0
SKIP_BUILD_DATASET=0
RUN_ABLATIONS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"; shift 2 ;;
    --config)
      CONFIG="$2"; shift 2 ;;
    --repo-root)
      REPO_ROOT="$2"; shift 2 ;;
    --clone-data)
      CLONE_DATA=1; shift ;;
    --skip-install)
      SKIP_INSTALL=1; shift ;;
    --skip-build-dataset)
      SKIP_BUILD_DATASET=1; shift ;;
    --run-ablations)
      RUN_ABLATIONS=1; shift ;;
    -h|--help)
      sed -n '1,40p' "$0"; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

if [[ -z "${HF_HOME:-}" ]]; then export HF_HOME="$REPO_ROOT/.cache/hf"; fi
if [[ -z "${TORCH_HOME:-}" ]]; then export TORCH_HOME="$REPO_ROOT/.cache/torch"; fi
if [[ -z "${XDG_CACHE_HOME:-}" ]]; then export XDG_CACHE_HOME="$REPO_ROOT/.cache/xdg"; fi
if [[ -z "${MPLCONFIGDIR:-}" ]]; then export MPLCONFIGDIR="$REPO_ROOT/.cache/mpl"; fi
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

if [[ $SKIP_INSTALL -eq 0 ]]; then
  .venv/bin/python -m pip install -U pip setuptools wheel
  .venv/bin/python -m pip install \
    numpy pandas pyyaml scikit-learn matplotlib pillow \
    torch torchvision timm
fi

if [[ $CLONE_DATA -eq 1 ]]; then
  mkdir -p data/external
  if [[ ! -d data/external/sample-data ]]; then
    git clone https://github.com/metrica-sports/sample-data data/external/sample-data
  fi
fi

if [[ $SKIP_BUILD_DATASET -eq 0 ]]; then
  PYTHONPATH=src .venv/bin/python -m soccer_vit.train build-dataset --config configs/default.yaml
fi

# 빠른 상태 확인
PYTHONPATH=src .venv/bin/python - <<'PY'
import torch, platform
print('platform:', platform.platform())
print('torch:', torch.__version__)
print('cuda_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu0:', torch.cuda.get_device_name(0))
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('mps available')
else:
    print('cpu only')
PY

case "$MODE" in
  smoke)
    # baseline 2종 + vit 스모크 + eval/report
    PYTHONPATH=src .venv/bin/python -m soccer_vit.train fit --model baseline_rule_like --config configs/default.yaml || true
    PYTHONPATH=src .venv/bin/python -m soccer_vit.train fit --model baseline_strict --config configs/default.yaml || true
    PYTHONPATH=src .venv/bin/python -m soccer_vit.train fit --model vit_base --config "$CONFIG"
    MPLBACKEND=Agg PYTHONPATH=src .venv/bin/python -m soccer_vit.eval run --config "$CONFIG"
    MPLBACKEND=Agg PYTHONPATH=src .venv/bin/python -m soccer_vit.report make --config "$CONFIG" --n-samples 8
    ;;
  train)
    PYTHONPATH=src .venv/bin/python -m soccer_vit.train fit --model vit_base --config "$CONFIG"
    ;;
  eval)
    MPLBACKEND=Agg PYTHONPATH=src .venv/bin/python -m soccer_vit.eval run --config "$CONFIG"
    MPLBACKEND=Agg PYTHONPATH=src .venv/bin/python -m soccer_vit.report make --config "$CONFIG" --n-samples 8
    ;;
  full)
    PYTHONPATH=src .venv/bin/python -m soccer_vit.train fit --model baseline_rule_like --config configs/default.yaml
    PYTHONPATH=src .venv/bin/python -m soccer_vit.train fit --model baseline_strict --config configs/default.yaml
    PYTHONPATH=src .venv/bin/python -m soccer_vit.train fit --model vit_base --config "$CONFIG"
    if [[ $RUN_ABLATIONS -eq 1 ]]; then
      PYTHONPATH=src .venv/bin/python -m soccer_vit.train fit --model vit_base --config configs/vit_mid_no_receiver.yaml
      MPLBACKEND=Agg PYTHONPATH=src .venv/bin/python -m soccer_vit.eval run --config configs/vit_mid_no_receiver.yaml
      PYTHONPATH=src .venv/bin/python -m soccer_vit.train fit --model vit_base --config configs/vit_mid_no_ball.yaml
      MPLBACKEND=Agg PYTHONPATH=src .venv/bin/python -m soccer_vit.eval run --config configs/vit_mid_no_ball.yaml
    fi
    MPLBACKEND=Agg PYTHONPATH=src .venv/bin/python -m soccer_vit.eval run --config "$CONFIG"
    MPLBACKEND=Agg PYTHONPATH=src .venv/bin/python -m soccer_vit.report make --config "$CONFIG" --n-samples 12
    ;;
  *)
    echo "Unknown mode: $MODE" >&2; exit 1 ;;
esac

echo "[done] mode=$MODE config=$CONFIG"
